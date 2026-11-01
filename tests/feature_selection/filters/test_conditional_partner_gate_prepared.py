"""The shared code matrix of the conditional partner gate gives the same verdicts as a gate rebuilt for every strong operand."""

import numpy as np

from mlframe.feature_selection.filters._mrmr_fe_step._conditional_maxt import ConditionalPartnerGate


def _data(n=6000, p=12, seed=0):
    """Ordinal codes: column 0 strong, column 1 adds information about y beyond it, the rest noise."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 6, size=(n, p)).astype(np.int8)
    y = ((data[:, 0] + data[:, 1]) % 3).astype(np.int64)
    return data, y


def test_shared_matrix_equals_fresh_gate_per_strong_operand():
    """Observed CMIs, the null floor and the clears() verdicts are equal for a reused gate and for a fresh gate per strong operand."""
    data, y = _data()
    cand = list(range(data.shape[1]))
    shared = ConditionalPartnerGate(data, y, cand, seed=3)
    for strong in (0, 2, 5):
        fresh = ConditionalPartnerGate(data, y, cand, seed=3)
        a = shared._compute(strong)
        b = fresh._compute(strong)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]
        for weak in cand:
            if weak != strong:
                assert shared.clears(weak, strong) == fresh.clears(weak, strong)
    assert shared._prepared() is shared._prepared()


def test_degenerate_target_is_remembered_as_none():
    """A constant target has no test; the verdict is cached and clears() keeps the benefit of the doubt."""
    data, _ = _data()
    gate = ConditionalPartnerGate(data, np.zeros(data.shape[0], dtype=np.int64), list(range(data.shape[1])))
    assert gate._prepared() is None and gate._prepared() is None
    assert gate.clears(1, 0) is True
