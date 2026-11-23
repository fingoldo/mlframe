"""Cluster 12 regression: the power-report quantile block survives an empty tau list (audit 2026-11-20, finding 12-F18).

With no finite tau the block used to crash on arr.max(). It now returns an explicit line; a populated list still renders the table.
"""

from mlframe.feature_selection._benchmarks.fs_hybrid._power import _quantile_block


def test_no_finite_tau_returns_a_plain_line():
    """All-non-finite taus give a stated 'no table' line and no exception (12-F18)."""
    out = _quantile_block([float("nan"), float("inf")], declared_r=5, effects=[0.1])
    assert any("No finite tau observed" in line for line in out)


def test_populated_taus_still_render_the_table():
    """Control: finite taus render the observed-tau summary and the planning table (12-F18)."""
    out = _quantile_block([0.1, 0.2, 0.3], declared_r=5, effects=[0.1])
    assert any(line.startswith("Observed tau over 3") for line in out)
    assert any(line.startswith("| median") for line in out)
