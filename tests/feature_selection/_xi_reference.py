"""A numpy-only reference of the Xi batch: ``lexsort`` by (value, counter-hash tie key, row), with the hash written out independently of the numba and CUDA kernels."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._orthogonal_xi_fe import _xi_from_order

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_G = np.uint64(0x9E3779B97F4A7C15)
_STRIDE = np.uint64(0xD1B54A32D192ED03)


def _mix(x):
    """splitmix64 finaliser on uint64 arrays or scalars."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        return x ^ (x >> np.uint64(31))


def tie_keys(random_state: int, j: int, n: int) -> np.ndarray:
    """The 32-bit tie key of every row for finite column number ``j`` of a batch seeded ``random_state``."""
    with np.errstate(over="ignore"):
        seed = _mix(np.uint64(random_state) * _STRIDE + np.uint64(j))
        rows = np.arange(n, dtype=np.uint64)
        return (_mix(seed + (rows + np.uint64(1)) * _G) >> np.uint64(32)).astype(np.uint64)


def xi_reference(X: np.ndarray, y: np.ndarray, random_state: int) -> np.ndarray:
    """Xi per column, finite columns numbered in order for the tie keys, non-finite columns 0."""
    n, k = X.shape
    out = np.zeros(k)
    j = 0
    rows = np.arange(n)
    for c in range(k):
        col = X[:, c]
        if not np.isfinite(col).all():
            continue
        order = np.lexsort((rows, tie_keys(random_state, j, n), col))
        out[c] = _xi_from_order(y[order])
        j += 1
    return out
