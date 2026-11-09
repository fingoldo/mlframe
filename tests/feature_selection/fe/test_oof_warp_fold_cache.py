"""The shared fold ids of ``fit_oof_warp1d`` equal the per-call permutation they replace."""

import numpy as np

from mlframe.feature_selection.filters._oof_warp_service import _fold_ids


def test_fold_ids_match_direct_permutation_and_are_shared():
    """The cached ids equal the direct permutation and the same array is returned on a repeat call."""
    for n, k, seed in [(1000, 5, 0), (777, 3, 7)]:
        direct = (np.random.default_rng(seed).permutation(n) % k).astype(np.int64)
        got = _fold_ids(n, k, seed)
        assert np.array_equal(got, direct)
        assert _fold_ids(n, k, seed) is got
