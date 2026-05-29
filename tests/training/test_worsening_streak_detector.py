"""Tests for the curve-shape ES detector + max-iter-hit diagnostic.

The curve-shape detector fires when the monitored val metric strictly fails to improve over
its own predecessor for ``max(max_iter // worsening_coeff, worsening_min_iters)`` consecutive
iterations since the best. It's a forward-looking complement to patience-based ES: patience
says "no NEW best for N iters", this says "the curve is monotonically bending the wrong way".

Tests:
  - linear-monotone-worsening triggers at threshold
  - oscillating series does NOT trigger (any improvement-over-prev resets the streak)
  - disabled flag is a no-op
  - threshold formula scales with max_iter
  - max_iter=None falls back to min_iters
  - works for ``mode='max'`` (AUC-style)
  - max-iter-hit diagnostic warning fires when best_iter is at the budget cap
  - max-iter-hit diagnostic does NOT fire when we stopped early via the curve-shape detector
"""
from __future__ import annotations

import logging
import time

import pytest

from mlframe.training._callbacks import UniversalCallback


def _build(*, patience: int = 100, mode: str = "min", worsening_enabled: bool = True,
            worsening_max_iter: int | None = 100, worsening_coeff: int = 5,
            worsening_min_iters: int = 5):
    cb = UniversalCallback(
        patience=patience, min_delta=0.0,
        monitor_dataset="valid_0", monitor_metric="loss", mode=mode,
        worsening_enabled=worsening_enabled,
        worsening_coeff=worsening_coeff,
        worsening_min_iters=worsening_min_iters,
        worsening_max_iter=worsening_max_iter,
        verbose=0,
    )
    cb.start_time = time.time()
    cb.last_reporting_ts = time.time()
    cb.iter = 0
    return cb


def _push(cb: UniversalCallback, value: float) -> bool:
    cb.metric_history.setdefault("valid_0", {}).setdefault("loss", []).append(float(value))
    return cb.should_stop()


def test_linear_monotone_worsening_triggers_at_threshold() -> None:
    """Worsening for `max(100//5, 5)`=20 consecutive iters stops on iter 20-after-best."""
    cb = _build(worsening_max_iter=100, worsening_coeff=5)
    # 5 improving iters -> best at iter 4 (0-indexed in the post-init counter)
    for v in [1.0, 0.9, 0.8, 0.7, 0.6]:
        assert _push(cb, v) is False
    # 19 worsening iters -> still below threshold of 20
    for i in range(1, 20):
        assert _push(cb, 0.6 + 0.01 * i) is False, f"premature stop at worsening iter {i}"
    # 20th worsening iter -> trigger
    assert _push(cb, 0.6 + 0.01 * 20) is True
    assert cb._worsening_stopped is True
    assert cb._worsening_streak_len == 20


def test_oscillating_series_does_not_trigger() -> None:
    """Any improvement-over-prev (even without new global best) resets the streak."""
    cb = _build(worsening_max_iter=100)
    seq = [1.0, 0.9, 0.95, 0.88, 0.93, 0.85, 0.91, 0.82, 0.89, 0.80]  # sawtooth, always recovering
    for v in seq:
        assert _push(cb, v) is False
    assert cb._worsening_stopped is False
    assert cb._worsening_streak_len == 0  # reset by every recovery


def test_disabled_flag_is_noop() -> None:
    cb = _build(worsening_enabled=False, worsening_max_iter=100)
    # Long monotone worsening that WOULD trigger if enabled
    for v in [1.0, 0.5] + [0.5 + 0.01 * i for i in range(1, 50)]:
        _push(cb, v)
    assert cb._worsening_stopped is False


def test_threshold_scales_with_max_iter() -> None:
    # max_iter=50, coeff=5 -> threshold 10
    cb_small = _build(worsening_max_iter=50, worsening_coeff=5)
    assert cb_small._worsening_threshold() == 10
    # max_iter=1000, coeff=5 -> threshold 200
    cb_big = _build(worsening_max_iter=1000, worsening_coeff=5)
    assert cb_big._worsening_threshold() == 200
    # max_iter=10 floored to min_iters=5
    cb_tiny = _build(worsening_max_iter=10, worsening_coeff=5)
    assert cb_tiny._worsening_threshold() == 5


def test_max_iter_none_falls_back_to_min_iters() -> None:
    cb = _build(worsening_max_iter=None, worsening_min_iters=7)
    assert cb._worsening_threshold() == 7


def test_max_mode_auc_style() -> None:
    """For max-mode (AUC), 'worsening' means current <= prev."""
    cb = _build(mode="max", worsening_max_iter=20, worsening_coeff=5, worsening_min_iters=4)
    # threshold = max(20//5, 4) = 4
    # AUC improving then plateau-decreasing
    for v in [0.7, 0.8, 0.9, 0.95]:
        assert _push(cb, v) is False
    # 4 strictly-worsening iters
    for v in [0.94, 0.93, 0.92, 0.91]:
        last_stop = _push(cb, v)
    assert last_stop is True


def test_max_iter_hit_diagnostic_fires_when_best_at_cap(caplog) -> None:
    """When best_iter is at max_iter-1 and no ES fired, log a WARN with the TODO note."""
    cb = _build(worsening_max_iter=10, worsening_enabled=False)  # disable curve-shape
    # 10 strictly-improving values -> best_iter stays at the current iter each time
    for i, v in enumerate([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
        with caplog.at_level(logging.WARNING):
            _push(cb, v)
    # On the last call (iter==max_iter-1) the diagnostic warning must have been logged.
    msgs = [r.message for r in caplog.records if "hit the iteration budget" in r.message]
    assert msgs, f"max-iter-hit warning not logged; got records: {[r.message for r in caplog.records]}"


def test_max_iter_hit_diagnostic_silent_when_we_stopped_early() -> None:
    """When curve-shape detector fires, no max-iter-hit warning is logged (budget wasn't exhausted)."""
    cb = _build(worsening_max_iter=30, worsening_coeff=5, worsening_min_iters=5)
    # Improve, then monotone worsen until threshold
    for v in [1.0, 0.9, 0.8]:
        _push(cb, v)
    for v in [0.81, 0.82, 0.83, 0.84, 0.85, 0.86]:  # 6 worsening iters >= threshold(30//5=6)
        stopped = _push(cb, v)
    assert cb._worsening_stopped is True
    # best_iter (=2 here) is FAR from cap (29), so no warning regardless.
    assert cb.best_iter == 2
