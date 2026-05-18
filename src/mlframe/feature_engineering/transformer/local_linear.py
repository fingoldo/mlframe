"""Local linear regression attention — fit OLS on each row's top-k neighbours and return regression coefficients as features.

Why this is fundamentally different from row-attention:
- Row-attention returns a weighted MEAN of neighbour targets (0th-order local estimate).
- Local linear regression returns the COEFFICIENTS of a local OLS fit (1st-order local estimate). The output is the per-feature gradient of y vs X computed locally.
- Boostings split on individual features but cannot compute local linear gradients natively — they would need many splits to approximate a smooth gradient.

For each query row, the output features are:
- ``β_0``: local intercept (analogous to y_mean but bias-corrected against features)
- ``β_1, ..., β_d``: local slopes (∂y/∂X_j at this query, estimated from k neighbours)
- ``r2``: how well the local linear model fits — high R² indicates a locally-smooth manifold; low R² indicates a decision boundary or noise region.

Output dimensionality: ``(d + 2)`` per row. For kin8nm d=8 → 10 features per row. Boostings can split on the slope coefficients to detect "regions where increasing X_3 increases y locally" — a structural pattern they can't synthesise from raw X alone.

Reference: locally-weighted regression / LOWESS (Cleveland 1979); local polynomial regression (Fan & Gijbels 1996).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._row_attention_ann import build_hnsw_index, query_topk
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_local_linear_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    k: int = 32,
    ridge_alpha: float = 1e-3,
    standardize: bool = True,
    return_r2: bool = True,
    column_prefix: str = "loclr",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Local linear regression features. For each row, find its top-k neighbours and fit ridge regression on (X_neighbours -> y_neighbours).

    Output features per row: [intercept, slope_per_input_column, optionally r2].
    Total cols = ``d + 1 + (1 if return_r2 else 0)``.

    Parameters:
        ``k`` - number of neighbours in each local fit. Must exceed d+1 for a non-degenerate fit.
        ``ridge_alpha`` - L2 regularisation to keep OLS stable when neighbours are nearly co-linear. Small value (1e-3) preserves the linear-regression interpretation while preventing singular matrices.
        ``standardize`` - RobustScaler on X before neighbour search and regression. Otherwise high-variance features dominate distances.
        ``return_r2`` - if True, append the local R² of the linear fit as the last feature.

    Mode A (X_query=None): OOF for X_train — each row's local fit uses non-self neighbours (per splitter folds).
    Mode B (X_query!=None): full-train bank.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
    n_train, d = X_train.shape
    if k < d + 2:
        raise ValueError(f"k={k} must be at least d+2={d+2} for a non-degenerate local fit.")

    # Standardise train; the standardiser is fit on train and applied to query — standard ML hygiene.
    if standardize:
        scaler = RobustScaler().fit(X_train)
        X_train_s = scaler.transform(X_train).astype(dtype, copy=False)
        X_query_s = scaler.transform(X_query).astype(dtype, copy=False) if X_query is not None else None
    else:
        X_train_s = X_train.astype(dtype, copy=False)
        X_query_s = X_query.astype(dtype, copy=False) if X_query is not None else None

    n_out_cols = d + 1 + (1 if return_r2 else 0)
    out_for_query = X_query is not None

    def _fit_and_extract(X_anchor: np.ndarray, X_neighbour_pool: np.ndarray, y_neighbour_pool: np.ndarray) -> np.ndarray:
        """For each anchor row, find k nearest in neighbour_pool, fit ridge, return [intercept, slopes, r2?]."""
        n_anchor = X_anchor.shape[0]
        out = np.zeros((n_anchor, n_out_cols), dtype=dtype)
        # Build ANN index over the neighbour pool.
        index = build_hnsw_index(X_neighbour_pool, space="cosine", M=16, ef_construction=100, num_threads=None)
        topk_ids, _ = query_topk(index, X_anchor, k=k)
        for q in range(n_anchor):
            ids = topk_ids[q]
            Xn = X_neighbour_pool[ids]
            yn = y_neighbour_pool[ids]
            model = Ridge(alpha=ridge_alpha, fit_intercept=True)
            try:
                model.fit(Xn, yn)
                out[q, 0] = model.intercept_
                out[q, 1 : 1 + d] = model.coef_
                if return_r2:
                    pred = model.predict(Xn)
                    ss_res = float(np.sum((yn - pred) ** 2))
                    ss_tot = float(np.sum((yn - yn.mean()) ** 2))
                    out[q, 1 + d] = 1.0 - ss_res / max(ss_tot, 1e-12)
            except Exception as exc:  # pragma: no cover - degenerate fits
                logger.info("local_linear: fit failed on row %d (%s); leaving zeros", q, exc)
        return out

    if not out_for_query:
        # Mode A: OOF for X_train.
        out = np.zeros((n_train, n_out_cols), dtype=dtype)
        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_train_s)):
            X_pool = X_train_s[tr_idx]
            y_pool = y_train[tr_idx].astype(np.float32, copy=False)
            X_anchor = X_train_s[va_idx]
            out[va_idx] = _fit_and_extract(X_anchor, X_pool, y_pool)
            logger.info("local_linear fold %d done (%d val rows)", fold_idx + 1, len(va_idx))
    else:
        # Mode B: single-pass.
        out = _fit_and_extract(X_query_s, X_train_s, y_train.astype(np.float32, copy=False))

    # Column names: intercept, slope per feature, r2.
    names = [f"{column_prefix}_intercept"]
    for j in range(d):
        names.append(f"{column_prefix}_slope_x{j}")
    if return_r2:
        names.append(f"{column_prefix}_r2")
    return pl.DataFrame({name: out[:, i] for i, name in enumerate(names)})
