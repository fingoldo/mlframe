"""The local gate's null permutations are generated once per row count and equal the fixed-seed generator output."""

import numpy as np

from mlframe.feature_selection.filters import _unified_fe_gate as ug


def test_permutations_equal_fixed_seed_generator_and_are_shared():
    """Same values as a fresh seed-0 generator, the same arrays on a repeat call, and read-only."""
    ug._PERMUTATION_MEMO.clear()
    got = ug.local_gate_permutations(1234)
    rng = np.random.default_rng(0)
    for p in got:
        assert np.array_equal(p, rng.permutation(1234))
    again = ug.local_gate_permutations(1234)
    assert all(a is b for a, b in zip(got, again)), "a repeat call must reuse the generated permutations"
    assert not got[0].flags.writeable
    assert len(got) == ug._LOCAL_GATE_NULL_PERMUTATIONS
