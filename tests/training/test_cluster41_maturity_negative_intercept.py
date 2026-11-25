"""Cluster 41 regression: a negative zero-window level is not a censoring verdict (audit 2026-11-20, finding 41-F31).

A monotone decline toward the present whose extrapolated level at age zero is negative used to be called "censoring_likely". Accrual from zero
does not produce a negative level, so that verdict is now reserved for a non-negative level near zero. The series has enough bins to get a
verdict at all, and a control with a small positive level keeps the censoring verdict.
"""

from mlframe.training.targets.target_maturity_audit import audit_binned_target_maturity


def test_negative_zero_window_level_is_not_censoring():
    """A declining series that extrapolates below zero is declining_level, not censoring_likely (41-F31)."""
    ages = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    stats = [2 * a - 1 for a in ages]
    result = audit_binned_target_maturity(bin_stats=stats, bin_ages=ages)
    assert result.verdict == "declining_level"


def test_small_positive_zero_window_level_stays_censoring():
    """Control: a declining series that extrapolates to a small positive level is still censoring_likely (41-F31)."""
    ages = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    stats = [a + 1 for a in ages]
    result = audit_binned_target_maturity(bin_stats=stats, bin_ages=ages)
    assert result.verdict == "censoring_likely"
