"""Linear-redundancy (masking) test shared by selectors that otherwise judge each feature on relevance alone.

A relevance test (contrast/shadow importance, marginal CV gain) passes every member of a collinear block: with an
exact identity ``x3 = 2*x1 - x2`` or a near-duplicate pair all members look relevant, and the selected subset ends up
with a singular or near-singular Gram. ``linear_r2`` measures how much of a candidate the already-kept columns
explain linearly; a candidate at R^2 >= 0.95 (VIF >= 20, the conventional severe-collinearity line) carries no
information the kept set lacks.
"""
from __future__ import annotations

import numpy as np

# R^2 above which a candidate counts as linearly masked by the kept columns. 0.95 == VIF 20.
DEFAULT_MASKING_R2 = 0.95


def linear_r2(kept: np.ndarray, candidate: np.ndarray) -> float:
    """R^2 of an intercept least-squares fit of ``candidate`` (n,) on ``kept`` (n, k), over rows finite in both.

    Returns 1.0 for a (finite-rows) constant candidate - an intercept model already carries it - and 0.0 when too
    few finite rows remain to fit."""
    kept = np.asarray(kept, dtype=np.float64)
    if kept.ndim == 1:
        kept = kept[:, None]
    cand = np.asarray(candidate, dtype=np.float64)
    rows = np.isfinite(cand) & np.all(np.isfinite(kept), axis=1)
    if rows.sum() <= kept.shape[1] + 2:
        return 0.0
    t = cand[rows] - cand[rows].mean()
    ss_tot = float(t @ t)
    scale = float(np.abs(cand[rows]).max()) if rows.any() else 0.0
    if ss_tot <= 1e-24 * max(1.0, scale) ** 2 * rows.sum():
        return 1.0
    A = kept[rows] - kept[rows].mean(axis=0)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    resid = t - A @ coef
    return 1.0 - float(resid @ resid) / ss_tot


def drop_linearly_masked(X: np.ndarray, accepted: np.ndarray, importance: np.ndarray, *, max_r2: float = DEFAULT_MASKING_R2) -> np.ndarray:
    """Walk accepted columns by descending ``importance`` (ties by column order) and drop each one the more important
    kept columns explain with R^2 >= ``max_r2``. Returns a new bool mask."""
    out = np.asarray(accepted, dtype=bool).copy()
    imp = np.nan_to_num(np.asarray(importance, dtype=np.float64), nan=-np.inf)
    order = sorted(np.nonzero(out)[0].tolist(), key=lambda j: (-float(imp[j]), j))
    kept: list[int] = []
    for j in order:
        if kept and linear_r2(X[:, kept], X[:, j]) >= max_r2:
            out[j] = False
        else:
            kept.append(j)
    return out


__all__ = ["DEFAULT_MASKING_R2", "linear_r2", "drop_linearly_masked"]
