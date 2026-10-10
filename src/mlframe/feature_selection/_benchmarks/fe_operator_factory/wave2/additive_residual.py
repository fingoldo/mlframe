"""Out-of-fold additive-model residual with a configurable number of bins per column (the pair-selection step of operator M needs no recipe: it stores nothing, only the chosen pairs).

``additive_oof_residual`` fits a binned backfit additive model (``nb`` quantile bins per column, ``sweeps`` backfit sweeps) on k-1 folds and predicts the held-out fold; the residual of every row is
therefore out of fold. ``bins_for(n)`` is the rule evaluated in ``m_bins``: more bins when there are more rows, so lack-of-fit of the additive part does not leak into the pair screen.
"""

from __future__ import annotations

import numpy as np

__all__ = ["additive_oof_residual", "bins_for"]


def bins_for(n: int, lo: int = 15, hi: int = 60, rows_per_bin: int = 250) -> int:
    """Bins per column for ``n`` fit rows: ``n / rows_per_bin`` clipped to ``[lo, hi]``."""
    return int(min(hi, max(lo, n // rows_per_bin)))


def _fit(X: np.ndarray, y: np.ndarray, nb: int, sweeps: int) -> tuple:
    """Backfit additive model on (X, y); returns (edges per column, tables per column, intercept)."""
    n, p = X.shape
    edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, nb + 1)[1:-1])) for j in range(p)]
    idx = [np.searchsorted(edges[j], X[:, j]) for j in range(p)]
    mu = float(y.mean())
    F = np.zeros((n, p))
    tabs = [np.zeros(len(edges[j]) + 1) for j in range(p)]
    for _ in range(sweeps):
        for j in range(p):
            r = y - mu - F.sum(1) + F[:, j]
            k = len(edges[j]) + 1
            t = np.bincount(idx[j], r, k) / np.maximum(np.bincount(idx[j], minlength=k), 1)
            t -= t.mean()
            F[:, j] = t[idx[j]]
            tabs[j] = t
    return edges, tabs, mu


def additive_oof_residual(X: np.ndarray, y: np.ndarray, nb: int = 15, sweeps: int = 3, k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold residual ``y - additive prediction`` for every row (k random disjoint folds)."""
    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    out = np.empty(n)
    for f in range(k):
        te = perm[f::k]
        m = np.ones(n, bool)
        m[te] = False
        edges, tabs, mu = _fit(X[m], y[m], nb, sweeps)
        pred = mu + sum(tabs[j][np.searchsorted(edges[j], X[te, j])] for j in range(X.shape[1]))
        out[te] = y[te] - pred
    return out
