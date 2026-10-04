"""The joint-gate operand floor rejects degraded copies of one operand but not pairs whose partner carries joint information."""

from __future__ import annotations

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_operand_floor import _beats_the_larger_operand

_MARGINALS = {0: 0.50, 1: 0.05}


def _marginal(var: int) -> float:
    """Fixed per-operand marginal MI table."""
    return _MARGINALS[var]


def test_noise_degraded_copy_of_strong_operand_is_rejected():
    """Engineered MI and the raw joint MI both sit ~1% above the strong operand, so the partner adds nothing: reject."""
    messages: list = []
    assert _beats_the_larger_operand(0.505, _marginal, (0, 1), messages, pair_mi=0.51) is False
    assert messages


def test_partner_with_real_joint_information_is_not_rejected():
    """The raw pair's joint MI is 1.5x the larger operand, so a low engineered uplift is not a degraded copy: accept."""
    assert _beats_the_larger_operand(0.52, _marginal, (0, 1), None, pair_mi=0.75) is True


def test_engineered_column_clearing_the_uplift_is_accepted():
    """An engineered MI 1.4x over the larger operand passes regardless of the joint MI."""
    assert _beats_the_larger_operand(0.70, _marginal, (0, 1), None, pair_mi=0.0) is True
