"""Per-column bin histograms for the artifact entropies, computed in one parallel pass.

The artifact builder needs ``H(X_j)`` for every retained feature, and took it one column at a time: a strided gather out of the shared binned
matrix, an ``np.bincount``, a mask and a log-sum, per feature, from Python. That is ``n_features`` independent reductions over one matrix, the
shape this codebase has repeatedly fused (see ``_extremality_matrix_njit``), and nothing in the artifact module was compiled at all.

Only the COUNTS are computed here. They are integers, so the parallel form gives exactly the same histogram as ``np.bincount``, and the
entropy itself stays in the caller's float code: moving a floating-point reduction into a hand-written loop would change its summation order
and with it the last bits of every reported ``H(X)`` and ``SU``, for no gain, since the log-sum is O(nbins) against the O(n) gather.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=True)
def _column_histograms_njit(data, col_indices, width):
    """``(len(col_indices), width)`` bin counts, one row per requested column of ``data``."""
    n_rows = data.shape[0]
    n_cols = col_indices.shape[0]
    out = np.zeros((n_cols, width), dtype=np.int64)
    for j in prange(n_cols):
        col = col_indices[j]
        for i in range(n_rows):
            out[j, data[i, col]] += 1
    return out


def column_histograms(data, col_indices, width: int):
    """Bin counts for the given columns of the binned matrix, as an ``(n_cols, width)`` integer array."""
    cols = np.ascontiguousarray(np.asarray(col_indices, dtype=np.int64))
    if cols.size == 0:
        return np.zeros((0, int(width)), dtype=np.int64)
    return _column_histograms_njit(np.ascontiguousarray(data), cols, int(width))
