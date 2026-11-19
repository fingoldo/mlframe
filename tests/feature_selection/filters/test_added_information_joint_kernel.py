"""The fused count kernel of the added-information test gives the p-value of the ``bincount`` form, which the float-candidate route still takes."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _added_information_gate as gate


def _codes(n=20000, kc=6, ks=4, ky=3, seed=0):
    """Candidate, source and target codes with a dependence of the target on both."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, ks, n)
    c = rng.integers(0, kc, n)
    y = ((s + c + rng.integers(0, 3, n)) % ky).astype(np.int64)
    return c, s, y


@pytest.mark.parametrize("with_source", [True, False])
def test_p_value_equals_the_float_candidate_route(with_source):
    """An integer candidate (the kernel) and the same values as floats (the unique + bincount route) give the same p-value, also with a gap in the labels."""
    c, s, y = _codes()
    c = c * 3 + 1  # occupied values 1, 4, 7, ...: the renumbering has to close the gaps
    src = s if with_source else None
    got = gate.added_information_p(c, src, y)
    want = gate.added_information_p(c.astype(np.float64), src, y)
    assert got == want and 0.0 <= got <= 1.0


def test_joint_counts_kernel_table():
    """The table is the count of every (source, renumbered candidate, target) triple."""
    c, s, y = _codes(5000)
    joint, k = gate._joint_counts_kernel(c * 2, s.astype(np.int64), y, 4, 3, 1 << 22)
    assert k == 6 and joint.shape == (4, 6, 3) and joint.sum() == 5000
    for i in (0, 3):
        for j in (0, 5):
            for t in (0, 2):
                assert joint[i, j, t] == np.count_nonzero((s == i) & (c == j) & (y == t))


def test_edge_cases_keep_their_answers():
    """A constant candidate has no degrees of freedom (1.0), a sparse table is NaN, a negative label takes the unique route and a short column is handled."""
    c, s, y = _codes(3000)
    assert gate.added_information_p(np.zeros_like(c), s, y) == 1.0
    assert np.isnan(gate.added_information_p(np.arange(3000), s, y))
    neg = c - 3
    assert gate.added_information_p(neg, s, y) == gate.added_information_p(neg.astype(np.float64), s, y)
    assert gate.added_information_p(c[:0], None, y[:0]) == 1.0
