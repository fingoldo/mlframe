"""The Fleuret permutation confidence is the same for any worker count (tracker 2026-11-02 wave 14, PAR14-2)."""

import numpy as np

from mlframe.feature_selection.filters.fleuret import get_fleuret_criteria_confidence_parallel


def _confidence(n_workers):
    """Confidence of a noise candidate with a gain bar that about half the permutations clear, with early stopping effectively off."""
    rng = np.random.default_rng(3)
    n = 1500
    cols = [rng.integers(0, 3, size=n).astype(np.int32) for _ in range(4)]
    data = np.column_stack(cols).astype(np.int32)
    nbins = np.array([3, 3, 3, 3], dtype=np.int64)
    budget = 60
    return get_fleuret_criteria_confidence_parallel(
        data_copy=data.copy(), factors_nbins=nbins, x=(1,), y=np.asarray([3], dtype=np.int64), selected_vars=[0], bootstrapped_gain=0.004,
        npermutations=budget, max_failed=budget + 1, nexisting=0, cached_cond_MIs={}, entropy_cache={}, n_workers=n_workers,
        parallel_kwargs={"backend": "threading"}, base_seed=11,
    )[1]


def test_one_and_three_workers_agree():
    """Per-worker seeds and a cumulative in-place shuffle made the permutation set depend on the split; it is now per permutation."""
    assert _confidence(1) == _confidence(3) == _confidence(4)
