"""eval_candidates on zero rows returns zero MI instead of reading sub[-1] of an empty sample (tracker 2026-11-02 wave 10, NJIT10 note)."""

import numpy as np

from mlframe.feature_selection.filters import _numba_utils_offset as off


def test_zero_rows_give_zero_scores():
    """With n=0 the edge sample is empty; both outputs are set to 0 rather than indexing past it."""
    p = 3
    Zt = np.zeros((p, 0))
    subsets = np.zeros((2, off.MAX_SUBSET), dtype=np.int64)
    sizes = np.array([2, 2], dtype=np.int64)
    stats = np.zeros(2, dtype=np.int64)
    out_even = np.full(2, 9.0)
    out_odd = np.full(2, 9.0)
    off.eval_candidates(Zt, subsets, sizes, stats, np.zeros(0, dtype=np.int64), 2, 4, out_even, out_odd)
    assert (out_even == 0.0).all() and (out_odd == 0.0).all()
