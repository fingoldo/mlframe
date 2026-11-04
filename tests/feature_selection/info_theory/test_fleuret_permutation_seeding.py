"""Permutation seeding and per-permutation column restore in ``get_fleuret_criteria_confidence``.

Each case is a concrete fixture whose verdict count changes under one of three mutants of the core: the default ``perm_offset`` becoming 1,
the per-permutation seed becoming ``perm_offset + index + 2``, and the per-permutation restore being removed. The expected counts are pinned
to the current implementation; a change to the seed stream or the restore order moves them.
"""

import numpy as np
from numba import types
from numba.typed import Dict as NumbaDict

from mlframe.feature_selection.filters._numba_utils import get_fleuret_criteria_confidence


def _fixture(seed: int, n: int):
    """Four binary columns, the last one a noisy copy of the first, so the target column carries a signal the permutations can destroy."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2, size=n).astype(np.int32) for _ in range(4)]
    cols[3] = (cols[0] ^ (rng.random(n) < 0.1)).astype(np.int32)
    return np.column_stack(cols).astype(np.int32), np.array([2, 2, 2, 2], dtype=np.int64)


def _cache():
    """A fresh numba dict for the conditional-MI and entropy caches, which the core mutates in place."""
    return NumbaDict.empty(key_type=types.unicode_type, value_type=types.float64)


def _call(data, nbins, *, base_seed, nperm, gain, extra_x_shuffling, perm_offset=None):
    """Run the core on a copy of ``data`` with the x=(1,), y=(3,) pair and return ``(nfailed, nchecked)``; omit ``perm_offset`` to use the default."""
    kwargs = {} if perm_offset is None else {"perm_offset": perm_offset}
    return get_fleuret_criteria_confidence(
        data_copy=data.copy(),
        factors_nbins=nbins,
        x=(1,),
        y=np.asarray([3], dtype=np.int64),
        selected_vars=[0],
        npermutations=nperm,
        bootstrapped_gain=gain,
        max_failed=nperm,
        nexisting=0,
        mrmr_relevance_algo="fleuret",
        mrmr_redundancy_algo="fleuret",
        max_veteranes_interactions_order=1,
        cached_cond_MIs=_cache(),
        entropy_cache=_cache(),
        extra_x_shuffling=extra_x_shuffling,
        base_seed=np.uint64(base_seed),
        **kwargs,
    )


def test_default_perm_offset_is_zero():
    """Omitting ``perm_offset`` runs the same permutation stream as ``perm_offset=0``; on this fixture offset 1 gives a different count."""
    data, nbins = _fixture(1, 200)
    default = _call(data, nbins, base_seed=777, nperm=40, gain=0.01, extra_x_shuffling=False)
    assert default == (6, 40)
    assert default == _call(data, nbins, base_seed=777, nperm=40, gain=0.01, extra_x_shuffling=False, perm_offset=0)


def test_permutation_seed_is_base_seed_stream_at_offset_plus_index():
    """Permutation ``perm_offset + i`` draws from the stream seeded ``perm_offset + i + 1``; a shifted seed changes the count on this fixture."""
    data, nbins = _fixture(1, 200)
    assert _call(data, nbins, base_seed=12345, nperm=40, gain=0.01, extra_x_shuffling=True, perm_offset=1) == (2, 40)


def test_each_permutation_starts_from_the_original_columns():
    """Without the per-permutation restore, later permutations shuffle already-shuffled columns and the count changes on this fixture."""
    data, nbins = _fixture(1, 300)
    assert _call(data, nbins, base_seed=12345, nperm=40, gain=0.02, extra_x_shuffling=True, perm_offset=0) == (1, 40)
