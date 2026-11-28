"""Cluster 41 regression: a diagnostic with its own cap never exceeds the block's remaining budget (audit 2026-11-20, finding 41-F35)."""

from mlframe.training.reporting._diagnostics_budget import DiagnosticsBudget


def test_cap_is_the_smaller_of_own_cap_and_remaining_budget():
    """With 100 s left the own 20 s cap stands; with 5 s left it shrinks to 5; a disabled budget leaves it alone."""
    b = DiagnosticsBudget(100.0, verbose=False)
    assert b.cap_for(20.0) == 20.0
    b.max_seconds = 5.0
    assert b.cap_for(20.0) <= 5.0
    b.max_seconds = 0.0
    assert b.cap_for(20.0) == 20.0
