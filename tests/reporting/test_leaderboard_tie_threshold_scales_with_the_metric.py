"""The leaderboard tie warning is relative to the metric's magnitude, not a fixed 5e-3."""

from __future__ import annotations

from mlframe.reporting.charts.model_comparison import _leaderboard_panel


def _title(a: float, b: float, higher_is_better: bool) -> str:
    """Title of the leaderboard for two models scoring ``a`` and ``b`` on one metric."""
    per_model = {"m1": {"metrics": {"score": a}}, "m2": {"metrics": {"score": b}}}
    return _leaderboard_panel(per_model, "score", higher_is_better, None).title


def test_large_scale_metric_is_not_called_tied_on_a_tiny_relative_gap_in_the_wrong_direction():
    """RMSE 10000 vs 10040 differ by 0.4%, a tie; 10000 vs 11000 are clearly apart even though both gaps exceed 5e-3 absolute."""
    assert "treat them as tied" in _title(10000.0, 10040.0, False)
    assert "treat them as tied" not in _title(10000.0, 11000.0, False)


def test_small_scale_metric_is_not_always_tied():
    """Log-losses of 1e-4 and 3e-4 differ by 200%, which the old absolute threshold called a tie."""
    assert "treat them as tied" not in _title(1e-4, 3e-4, False)
