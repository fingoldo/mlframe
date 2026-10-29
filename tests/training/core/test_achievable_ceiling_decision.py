"""Fast pure-arithmetic checks of the proceed/skip branches in ``_ceiling_decision`` (no model fits)."""

from mlframe.training.core._achievable_ceiling import _ceiling_decision


def _call(*, floor, composite, std, margin=0.05, frac=0.5):
    """Run ``_ceiling_decision`` on a no-lag verdict whose floor is the raw RMSE."""
    common = dict(raw_rmse=floor, lag_rmse=float("nan"), best_composite_rmse=composite, best_base=None,
                  floor_rmse=floor, margin=margin, n_fit=100, n_hold=50, lag1_ar=None)
    return _ceiling_decision(common, raw_rmse=floor, lag_rmse=float("nan"), best_composite_rmse=composite,
                             floor_rmse=floor, margin=margin, y_hold_std=std, strong_floor_frac=frac)


def test_proceed_when_composite_beats_floor_by_margin():
    """A composite that beats the floor by more than the margin proceeds with its headroom recorded."""
    verdict = _call(floor=1.0, composite=0.5, std=2.0)
    assert verdict["decision"] == "proceed"
    assert verdict["headroom_vs_min"] == 0.5


def test_skip_when_strong_floor_and_composite_cannot_beat_it():
    """A strong floor that the composite cannot beat by the margin skips lag-free discovery."""
    verdict = _call(floor=0.2, composite=0.2, std=2.0)
    assert verdict["decision"] == "skip"


def test_proceed_when_floor_is_weak_even_if_composite_cannot_beat_it():
    """A weak floor keeps discovery running even when the composite cannot beat it (low-confidence verdict)."""
    verdict = _call(floor=1.9, composite=1.9, std=2.0)
    assert verdict["decision"] == "proceed"
    assert "low-confidence" in verdict["reason"]
