"""Drop raw survivors that are exact linear combinations of higher-ranked survivors.

A raw column that the selection already spans (``x3 = 2*x1 - x2`` once ``x1`` and ``x2`` are kept) adds no information
to a linear model and makes the kept block's Gram matrix singular. Greedy Gram-Schmidt in relevance order keeps the
first representative of every such block and drops the rest.
"""

from __future__ import annotations

import numpy as np

__all__ = ["exact_dependence_keep_mask"]


def exact_dependence_keep_mask(X_num: np.ndarray, rel_tol: float = 1e-9) -> np.ndarray:
    """Return a boolean keep-mask over the columns of ``X_num``, in the given order.

    A column is dropped when its centred residual after projecting onto the already-kept columns is below
    ``rel_tol`` times its own centred norm. Zero-variance columns are kept untouched: they are not a linear
    combination of anything this guard can judge.
    """
    n_cols = X_num.shape[1]
    keep = np.ones(n_cols, dtype=bool)
    basis = np.empty((X_num.shape[0], 0), dtype=np.float64)
    for j in range(n_cols):
        v = X_num[:, j] - X_num[:, j].mean()
        norm_v = float(np.linalg.norm(v))
        if norm_v == 0.0:
            continue
        if basis.shape[1]:
            residual = v - basis @ (basis.T @ v)
        else:
            residual = v
        norm_r = float(np.linalg.norm(residual))
        if norm_r <= rel_tol * norm_v:
            keep[j] = False
            continue
        basis = np.column_stack([basis, residual / norm_r])
    return keep
