"""Shared pairwise-squared-euclidean-distance GEMM helper for the anchor/attention transformer
family (``class_conditional_anchor.py`` / ``anchor_attention.py`` / ``inducing_attention.py``):
independently duplicated across those modules, consolidated here so a fix can't silently drift
out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def squared_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared euclidean distance, (len(A), len(B)), via the ``||a||^2 - 2 a.b + ||b||^2`` GEMM
    decomposition. Avoids the (len(A), len(B), d) broadcast cube that ``((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1)``
    materialises; only the (len(A), len(B)) result is allocated. Differs from the subtraction form by float32 reduction
    order (~1e-5 to ~2e-7 relative on the downstream softmax/argmin), selection-equivalent for these FE/attention features."""
    a_sq = np.einsum("ij,ij->i", A, A)[:, None]
    b_sq = np.einsum("ij,ij->i", B, B)[None, :]
    d = a_sq - 2.0 * (A @ B.T) + b_sq
    np.maximum(d, 0.0, out=d)
    return np.asarray(d)
