"""Permutation-null MI for one column, all permutations at once, from permutations drawn up front.

The auto-base null block-shuffles each feature's bin codes ``auto_base_null_perms`` times and takes the MI against y's
codes, one Python iteration per (column, permutation): 481 us each at n=100k, serial. The permutations themselves are
cheap to draw, so they are drawn first in exactly the order the loop consumed the generator, and this kernel runs the
gather + joint-histogram MI for every permutation of a column in parallel. It calls the same gather and MI kernels the
loop called, so each null MI is the identical number.
"""

from __future__ import annotations

import numpy as np

try:
    import numba as _numba

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is a hard dependency in practice
    _numba = None
    _HAS_NUMBA = False

from ._collinear_numba import block_shuffle_gather
from ._screening_mi_pair import _mi_from_binned_pair

if _HAS_NUMBA:
    from ._collinear_numba import _block_gather_kernel
    from ._screening_mi_pair import _mi_from_binned_pair_njit_kernel

    @_numba.njit(cache=True, parallel=True, nogil=True)
    def _null_mis_kernel(col_codes, y_codes, perms, block_len, nbins):
        """MI of ``y_codes`` against each block-permuted copy of ``col_codes``, one permutation per parallel iteration."""
        n_perms = perms.shape[0]
        out = np.empty(n_perms, dtype=np.float64)
        for p in _numba.prange(n_perms):
            shuffled = _block_gather_kernel(col_codes, perms[p], block_len)
            out[p] = _mi_from_binned_pair_njit_kernel(shuffled, y_codes, nbins)  # type: ignore[misc]  # redefined to a real njit kernel under `if _HAS_NUMBA:`; only None on the branch this block never runs on
        return out


def draw_null_permutations(rng: np.random.Generator, n_rows: int, n_perms: int, block_len: int) -> np.ndarray:
    """The ``(n_perms, n_blocks)`` permutations the shuffle loop would draw for one column, consuming ``rng`` identically.

    With ``block_len <= 1`` the loop shuffled the elements themselves; ``rng.permutation(n)`` makes the same draws and
    gathering by it reproduces that shuffle, so it is represented here as blocks of length one.
    """
    n_blocks = n_rows if block_len <= 1 else (n_rows + block_len - 1) // block_len
    perms = np.empty((n_perms, n_blocks), dtype=np.int64)
    for p in range(n_perms):
        perms[p] = rng.permutation(n_blocks)
    return perms


def shuffle_by(arr: np.ndarray, perm: np.ndarray, block_len: int) -> np.ndarray:
    """``arr`` block-permuted by one pre-drawn permutation (element-permuted when ``block_len <= 1``)."""
    return block_shuffle_gather(arr, perm, max(1, int(block_len)))


def null_mis_binned(col_codes: np.ndarray, y_codes: np.ndarray, perms: np.ndarray, block_len: int, nbins: int) -> np.ndarray:
    """The null MI of every permutation in ``perms`` for one binned column, in parallel when numba is available."""
    bl = max(1, int(block_len))
    if _HAS_NUMBA:
        return np.asarray(_null_mis_kernel(
            np.ascontiguousarray(col_codes, dtype=np.int64), np.ascontiguousarray(y_codes, dtype=np.int64),
            np.ascontiguousarray(perms, dtype=np.int64), bl, int(nbins),
        ))
    return np.array([_mi_from_binned_pair(block_shuffle_gather(col_codes, perms[p], bl), y_codes, nbins=nbins) for p in range(perms.shape[0])])
