"""The family-wise partner test passes a real partner of a strong operand and none of a hundred noise columns."""

import numpy as np

from mlframe.feature_selection.filters._mrmr_fe_step._conditional_maxt import ConditionalPartnerGate


def _codes(n=6000, n_noise=100, seed=0):
    """Code matrix ``[strong, real partner, noise...]`` and a target that depends on both of the first two columns."""
    rng = np.random.default_rng(seed)
    strong = rng.integers(0, 8, n)
    partner = rng.integers(0, 4, n)
    y = (strong + 2 * partner) % 4
    flip = rng.random(n) < 0.1
    y = np.where(flip, rng.integers(0, 4, n), y)
    noise = rng.integers(0, 10, (n, n_noise))
    return np.column_stack([strong, partner, noise]).astype(np.int16), y.astype(np.int64)


def test_real_partner_clears_and_no_noise_column_does():
    """The partner that shapes the target given the strong operand clears the maximum null; none of the 100 noise columns does."""
    data, y = _codes()
    gate = ConditionalPartnerGate(data, y, list(range(data.shape[1])), seed=1)
    assert gate.clears(1, 0)
    passed = [j for j in range(2, data.shape[1]) if gate.clears(j, 0)]
    assert passed == [], f"noise columns that cleared the family-wise test: {passed}"


def test_constant_target_gives_no_verdict_against_the_pair():
    """With a constant target there is no information to condition on, so the test leaves a pair's verdict as it was."""
    data, _ = _codes()
    gate = ConditionalPartnerGate(data, np.zeros(data.shape[0], dtype=np.int64), list(range(data.shape[1])), seed=1)
    assert gate.clears(1, 0) and gate.clears(5, 0)


def test_unknown_partner_gives_no_verdict_against_the_pair():
    """A column outside the candidate list cannot be judged, so it keeps the verdict it had."""
    data, y = _codes()
    gate = ConditionalPartnerGate(data, y, [0, 1, 2], seed=1)
    assert gate.clears(99, 0)
