"""Unit tests for the shared MonotonicDeclineStopper detector.

Covers the exact streak contract: climb -> peak -> N consecutive strict declines fires;
bounce-up resets; plateau resets; new global best resets; min vs max mode; disabled;
NaN/inf no-op; latched stop.
"""

from __future__ import annotations

import pytest

from mlframe.estimators.early_stopping_monotonic import MonotonicDeclineStopper


def _feed(stopper, values):
    """Feed a sequence, return the 0-based index of the first True (or None)."""
    for i, v in enumerate(values):
        if stopper.update(v):
            return i
    return None


def test_climb_peak_then_N_decline_fires_max() -> None:
    s = MonotonicDeclineStopper(patience=3, mode="max")
    # climb to peak at 0.9, then 3 strict declines
    idx = _feed(s, [0.5, 0.7, 0.9, 0.88, 0.86, 0.84])
    assert idx == 5  # third decline (0.84) is the 3rd consecutive
    assert s.streak == 3
    assert s.stopped is True


def test_does_not_fire_before_N_declines() -> None:
    s = MonotonicDeclineStopper(patience=3, mode="max")
    idx = _feed(s, [0.5, 0.9, 0.88, 0.86])  # only 2 declines
    assert idx is None
    assert s.streak == 2
    assert s.stopped is False


def test_bounce_up_resets_streak() -> None:
    s = MonotonicDeclineStopper(patience=3, mode="max")
    # decline twice, bounce up (not a new best), decline again -> streak restarts
    idx = _feed(s, [0.5, 0.9, 0.88, 0.86, 0.87, 0.85, 0.84])
    assert idx is None  # the 0.87 bounce reset; only 2 declines after
    assert s.streak == 2


def test_plateau_resets_streak() -> None:
    s = MonotonicDeclineStopper(patience=3, mode="max")
    # decline twice, plateau (equal to prev), decline once more -> not yet 3-in-a-row
    idx = _feed(s, [0.5, 0.9, 0.88, 0.86, 0.86, 0.84])
    assert idx is None
    assert s.streak == 1


def test_new_global_best_resets_streak() -> None:
    s = MonotonicDeclineStopper(patience=3, mode="max")
    # two declines, then a NEW best above 0.9 (resets streak), then 3 declines -> fires after new run
    idx = _feed(s, [0.9, 0.85, 0.80, 0.95, 0.90, 0.88, 0.86])
    assert idx == 6
    assert s.best == 0.95


def test_min_mode_fires_on_rising_loss() -> None:
    s = MonotonicDeclineStopper(patience=3, mode="min")
    # loss falls to 0.1 then rises 3 times (worsening for min-mode)
    idx = _feed(s, [0.5, 0.3, 0.1, 0.12, 0.14, 0.16])
    assert idx == 5
    assert s.best == 0.1


def test_min_mode_plateau_and_bounce_reset() -> None:
    s = MonotonicDeclineStopper(patience=2, mode="min")
    # rise once, drop (bounce-down = improvement-over-prev), rise once -> no fire
    idx = _feed(s, [0.1, 0.2, 0.15, 0.18])
    assert idx is None


def test_disabled_never_fires() -> None:
    for p in (None, 0, -1):
        s = MonotonicDeclineStopper(patience=p, mode="max")
        assert s.enabled is False
        assert _feed(s, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]) is None


def test_nan_inf_are_noop() -> None:
    s = MonotonicDeclineStopper(patience=2, mode="max")
    s.update(0.9)
    s.update(0.88)  # streak 1
    assert s.update(float("nan")) is False
    assert s.update(float("inf")) is False
    assert s.streak == 1  # untouched by the non-finite values
    assert s.update(0.86) is True  # 2nd real decline fires


def test_nan_inf_logs_once(caplog) -> None:
    """Regression test for audit F6: a non-finite score used to be a fully silent no-op.

    Before the fix, feeding NaN/inf produced zero diagnostic signal that anything was wrong -- a run scoring
    NaN on every iteration looked identical to ordinary non-convergence. Verify a warning fires exactly once
    (rate-limited), not once per non-finite call.
    """
    import logging

    s = MonotonicDeclineStopper(patience=2, mode="max")
    s.update(0.9)
    with caplog.at_level(logging.WARNING, logger="mlframe.estimators.early_stopping_monotonic"):
        s.update(float("nan"))
        s.update(float("inf"))
        s.update(float("-inf"))
    warnings = [r for r in caplog.records if "non-finite" in r.message]
    assert len(warnings) == 1


def test_stop_is_latched() -> None:
    s = MonotonicDeclineStopper(patience=2, mode="max")
    _feed(s, [0.9, 0.8, 0.7])
    assert s.stopped is True
    # even a brilliant new value keeps the latched stop True
    assert s.update(2.0) is True


def test_reset_clears_state() -> None:
    s = MonotonicDeclineStopper(patience=2, mode="max")
    _feed(s, [0.9, 0.8, 0.7])
    s.reset()
    assert s.stopped is False
    assert s.best is None
    assert s.streak == 0
    assert _feed(s, [0.9, 0.8, 0.7]) == 2


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        MonotonicDeclineStopper(patience=3, mode="bogus")
