"""Spectral seriation: reorder a similarity / correlation matrix to surface block structure.

From PZAD «многомерный анализ» (slide 39): a raw correlation heatmap
``plt.imshow(df.corr())`` is unreadable because the features are in arbitrary order;
reordering rows/columns by a spectral score (the lecture uses ``svds(cr, k=1)`` +
``argsort``) groups correlated blocks together so the structure becomes visible.

This is reusable beyond plotting: the same permutation groups a feature-redundancy /
similarity matrix into blocks, which is useful for feature clustering, block-diagonal
detection, and ordering a DCD / co-occurrence matrix for inspection.

Two orderings:
- ``method='fiedler'`` (default): the Fiedler vector — eigenvector of the graph
  Laplacian for the 2nd-smallest eigenvalue of the affinity ``|M|``. The principled
  seriation ordering; minimizes the sum of |i-j| over strongly-connected pairs.
- ``method='svd'``: the lecture's leading-singular-vector order. Cheaper, but for an
  all-positive similarity the leading (Perron) vector separates blocks less cleanly
  than the Fiedler vector; kept for parity with the lecture.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["spectral_seriation", "seriate"]


def spectral_seriation(M: np.ndarray, *, method: str = "fiedler") -> np.ndarray:
    """Return a permutation of ``M``'s rows/columns that groups similar entries into contiguous blocks.

    Parameters
    ----------
    M : np.ndarray
        Square similarity / correlation matrix (need not be symmetric; it is symmetrized as ``(M + M.T)/2``).
    method : {'fiedler', 'svd'}
        Ordering score (see module docstring).

    Returns
    -------
    np.ndarray
        Integer permutation ``perm`` such that ``M[perm][:, perm]`` is seriated.
    """
    if method not in ("fiedler", "svd"):
        raise ValueError(f"spectral_seriation: method must be 'fiedler' or 'svd', got {method!r}.")
    M = np.ascontiguousarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"spectral_seriation: M must be square 2-D, got shape {M.shape}.")
    n = M.shape[0]
    if n <= 2:
        return np.arange(n)
    S = 0.5 * (M + M.T)
    if method == "svd":
        # Leading eigenvector of the symmetrized matrix (== leading singular vector for symmetric S).
        vals, vecs = np.linalg.eigh(S)
        lead = vecs[:, int(np.argmax(vals))]
        return np.argsort(lead)
    # Fiedler: affinity |S| -> Laplacian -> eigenvector of the 2nd-smallest eigenvalue.
    A = np.abs(S)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    vals, vecs = np.linalg.eigh(L)
    order_by_eigval = np.argsort(vals)
    # 1st eigenvector is the trivial constant (eigenvalue ~0); the Fiedler vector is the next one.
    fiedler = vecs[:, order_by_eigval[1]]
    return np.argsort(fiedler)


def seriate(M: np.ndarray, *, method: str = "fiedler", perm: np.ndarray | None = None):
    """Reorder ``M`` by spectral seriation. Returns ``(M_reordered, perm)``.

    Pass a precomputed ``perm`` to apply the same ordering to another matrix (e.g. reorder both a correlation
    matrix and its p-value matrix consistently).
    """
    M = np.ascontiguousarray(M, dtype=np.float64)
    if perm is None:
        perm = spectral_seriation(M, method=method)
    return M[np.ix_(perm, perm)], perm
