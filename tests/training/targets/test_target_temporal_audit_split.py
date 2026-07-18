"""Wave 106 (2026-05-21): split target_temporal_audit.py (1507 -> 933)
into 3 siblings: change-point detection, plot, and _audit_from_agg.
"""

from __future__ import annotations

from pathlib import Path


def test_moved_symbols_still_importable() -> None:
    """Moved symbols still importable."""
    from mlframe.training.targets.target_temporal_audit import (
        find_change_points_pelt,
        find_change_points_zscore,
        find_change_points,
        _segments_from_change_points,
        plot_target_over_time,
        _audit_from_agg,
    )

    for fn in (
        find_change_points_pelt,
        find_change_points_zscore,
        find_change_points,
        _segments_from_change_points,
        plot_target_over_time,
        _audit_from_agg,
    ):
        assert callable(fn), fn


def test_other_symbols_still_importable() -> None:
    """Other symbols still importable."""
    from mlframe.training.targets.target_temporal_audit import (
        audit_target_over_time,
        audit_targets_over_time,
        format_temporal_audit_report,
        TimeBin,
        TemporalAuditResult,
    )

    assert callable(audit_target_over_time)
    assert callable(audit_targets_over_time)
    assert callable(format_temporal_audit_report)
    assert TimeBin is not None
    assert TemporalAuditResult is not None


def test_facade_below_1k_line_threshold() -> None:
    """Facade below 1k line threshold."""
    root = Path(__file__).resolve().parents[3] / "src" / "mlframe" / "training" / "targets"
    facade = root / "target_temporal_audit.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"target_temporal_audit.py is {n} lines, still over the 1k threshold"


def test_changepoint_module_identity() -> None:
    """Changepoint module identity."""
    from mlframe.training.targets import target_temporal_audit, _target_temporal_changepoint

    assert target_temporal_audit.find_change_points_pelt is _target_temporal_changepoint.find_change_points_pelt
