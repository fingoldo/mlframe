"""Self-supervised residual row-attention — captures what an auxiliary boosting misses in each row's neighbourhood.

Pipeline:
1. Fit a small LightGBM on `(X_train, y_train)` via KFold to get OOF predictions ``y_hat_train_oof``.
2. Compute residuals: ``residual_train = y_train - y_hat_train_oof`` (regression) or ``y_train - p_hat_oof`` (binary).
3. Run row-attention with ``residual_train`` as the target.

Output features per row: "weighted mean of neighbour RESIDUALS" — i.e., "in my neighbourhood, what error pattern is the auxiliary boosting making?"

Why this should help boostings downstream:
- Plain row-attention on raw y produces "neighbour mean of y" — boosting can sometimes approximate this with enough splits.
- Residual row-attention produces "neighbour mean of what was MISSED by a previous boosting" — by definition the downstream boosting starts from no information about residual structure (it's a feature engineered FROM a boosting fit, not predictable WITHIN one boosting iteration). This is the canonical "stacking" / "boosting-of-boosting" pattern lifted to a neighbourhood-aggregate.

Leakage discipline:
- Auxiliary LGB is fit via KFold OOF to get unbiased residuals for train rows.
- Mode A (X_query=None): outer row-attention also uses its own splitter for OOF; the inner auxiliary OOF and the outer attention OOF can share the same splitter (we do) so each train row attends only to non-self residuals.
- Mode B (X_query!=None): auxiliary LGB is fit once on full train → y_hat for X_query (residual unknown for X_query, that's fine — we're using train residuals as the bank's "target"); outer attention uses full train residual_bank.

Reference: stacking (Wolpert 1992); gradient boosting + residual learning (Friedman 2001); FB GBDT+LR but with neighbour aggregate as the "LR" output.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from ._utils import require_seed, validate_numeric_input
from .row_attention import compute_row_attention

logger = logging.getLogger(__name__)


def compute_residual_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    task: str = "auto",
    aux_n_estimators: int = 100,
    aux_max_depth: int = 5,
    aux_learning_rate: float = 0.05,
    aux_n_splits: int = 5,
    n_heads: int = 4,
    head_dim: int = 8,
    k: int = 32,
    aggregate: tuple[str, ...] = ("y_mean", "y_std"),
    projection: Literal["random", "pls"] = "pls",
    column_prefix: str = "resid",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Residual row-attention: fits auxiliary LGB, takes OOF residuals, runs row-attention with residuals as target.

    Returns a polars DataFrame:
        - Mode A (``X_query is None``): OOF residual-attention features for X_train.
        - Mode B (``X_query is not None``): residual-attention features for X_query (using full-train residual bank).

    Output column names: ``{column_prefix}_h{head}_{agg}`` per the row-attention convention. Aggregate names with the "y_" prefix in the underlying call still
    refer to the y-target field, which is actually "residual" in this context. We keep the naming for consistency with the base API; users reading the output
    should mentally substitute "residual" for "y".

    Parameters specific to residual extraction:
        ``task``               - "auto" (infer from y), "regression", or "binary". Auto: binary iff y has exactly 2 unique values.
        ``aux_n_estimators``   - trees in the auxiliary LGB. 100 is enough for a smooth-target fit; more would overfit and shrink residuals to zero.
        ``aux_max_depth``      - depth cap; 5 is a typical small-LGB depth.
        ``aux_learning_rate``  - 0.05 is conservative; matches the downstream defaults to make residuals "what a similar LGB would have missed".
        ``aux_n_splits``       - KFold for auxiliary OOF predictions.

    Remaining kwargs (``n_heads, head_dim, k, aggregate, projection, ...``) forward directly to ``compute_row_attention``.
    """
    import lightgbm as lgb

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError(f"y_train length {y_train.shape[0]} != X_train rows {X_train.shape[0]}.")

    if task == "auto":
        unique_y = np.unique(y_train[~np.isnan(y_train)] if y_train.dtype.kind == "f" else y_train)
        task = "binary" if len(unique_y) == 2 else "regression"

    def _make_aux():
        common = dict(
            n_estimators=aux_n_estimators, max_depth=aux_max_depth, learning_rate=aux_learning_rate,
            random_state=seed, verbose=-1, n_jobs=-1, num_leaves=min(2 ** aux_max_depth, 63),
        )
        return lgb.LGBMClassifier(**common) if task == "binary" else lgb.LGBMRegressor(**common)

    # Step 1: OOF predictions on X_train via auxiliary LGB.
    y_hat_oof = np.zeros(X_train.shape[0], dtype=np.float32)
    aux_splitter = KFold(n_splits=aux_n_splits, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, va_idx) in enumerate(aux_splitter.split(X_train)):
        model = _make_aux()
        model.fit(X_train[tr_idx], y_train[tr_idx])
        if task == "binary":
            y_hat_oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1].astype(np.float32, copy=False)
        else:
            y_hat_oof[va_idx] = model.predict(X_train[va_idx]).astype(np.float32, copy=False)
    residual_train = (y_train.astype(np.float32) - y_hat_oof).astype(np.float32)
    logger.info("residual_attention: aux LGB OOF residuals: mean=%.4f, std=%.4f, abs_mean=%.4f", residual_train.mean(), residual_train.std(), np.abs(residual_train).mean())

    # Step 2: run row-attention with residuals as the "target".
    out = compute_row_attention(
        X_train=X_train, y_train=residual_train, X_query=X_query, splitter=splitter,
        seed=seed, n_heads=n_heads, head_dim=head_dim, k=k,
        aggregate=aggregate, projection=projection,
        gpu_stage4=False, dedupe_threshold=None, column_prefix=column_prefix,
    )
    return out
