"""Local regression-manifold curvature via quadratic fit on K=40 neighbors.

Iter 77 mechanism. Geometric agent's #5 ranked.

For each query row: fit local quadratic regression y_neighbor = a + b'(x_neighbor - x_query) +
0.5 (x_neighbor - x_query)' H (x_neighbor - x_query) using K=40 nearest train rows.

Emit:
- trace(H) — scalar mean curvature
- frobenius_norm(H) — total curvature magnitude
- linear_residual - quadratic_residual — improvement from adding quadratic term
- linear_fit_value — local linear y estimate
- quadratic_fit_value — local quadratic y estimate

5 features. Continuous-curvature analog of iter 69 baseline-disagreement (which is discrete model-class
disagreement). Same family as iter 72 (also input-X geometry) but captures shape rather than density.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_local_curvature_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    k_neighbors: int = 40,
    standardize: bool = True,
    column_prefix: str = "curv",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Local quadratic-fit curvature features per query row."""
    from sklearn.neighbors import NearestNeighbors

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        d = Xt_s.shape[1]
        k_eff = min(k_neighbors, Xt_s.shape[0])
        nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1).fit(Xt_s)
        _, idx = nn.kneighbors(Xq_s)
        n_q = Xq_s.shape[0]
        out = np.zeros((n_q, n_features), dtype=np.float32)
        # Upper-triangular (i <= j) index pairs for the quadratic cross-terms.
        # The index structure is loop-invariant across query rows, so it is
        # hoisted out of the per-row loop and the cross-terms / Hessian scatter
        # are built with a single broadcast instead of nested Python loops +
        # per-row column_stack (bit-identical column order and values).
        iu, ju = np.triu_indices(d)
        diag_mask = iu == ju
        ones_col = np.ones((k_eff, 1), dtype=np.float32)
        for q in range(n_q):
            nbr_X = Xt_s[idx[q]]  # (k_eff, d)
            nbr_y = y_t[idx[q]].astype(np.float32)
            dx = nbr_X - Xq_s[q]  # (k_eff, d)
            # Linear basis: [1, dx_1, dx_2, ..., dx_d]
            A_lin = np.concatenate([ones_col, dx], axis=1)
            try:
                coef_lin, _, _, _ = np.linalg.lstsq(A_lin, nbr_y, rcond=None)
                pred_lin = A_lin @ coef_lin
                resid_lin = float(np.sum((nbr_y - pred_lin) ** 2))
                # Quadratic basis: [linear basis, dx_i * dx_j for i <= j]
                quad = dx[:, iu] * dx[:, ju]  # (k_eff, d*(d+1)/2)
                A_quad = np.concatenate([A_lin, quad], axis=1)
                coef_quad, _, _, _ = np.linalg.lstsq(A_quad, nbr_y, rcond=None)
                pred_quad = A_quad @ coef_quad
                resid_quad = float(np.sum((nbr_y - pred_quad) ** 2))
                # Build H from quad coefficients
                # Quad coefs index: linear has 1 + d coefs; then quad coefs in order (i, j) for i <= j
                quad_coefs = coef_quad[1 + d :]
                # Off-diagonal entries get the raw coef on both sides; diagonal
                # entries get 2*coef (second derivative = 2*a_ii for the x_i^2 coef).
                H = np.zeros((d, d), dtype=np.float32)
                H[iu, ju] = quad_coefs
                H[ju, iu] = quad_coefs
                H[iu[diag_mask], ju[diag_mask]] = 2.0 * quad_coefs[diag_mask]
                trace_H = float(np.trace(H))
                frob_H = float(np.sqrt(np.sum(H**2)))
                # Predict at query point (dx = 0): value is the intercept of linear/quadratic.
                lin_val = float(coef_lin[0])
                quad_val = float(coef_quad[0])
                resid_diff = resid_lin - resid_quad  # positive = quadratic fits better
                out[q, 0] = trace_H
                out[q, 1] = frob_H
                out[q, 2] = resid_diff
                out[q, 3] = lin_val
                out[q, 4] = quad_val
            except Exception:
                # Singular matrix or other numerical issue; use zeros
                out[q] = 0.0
        return out

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_trace_H"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_frob_H"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_resid_diff"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_lin_val"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_quad_val"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("local_curvature: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
