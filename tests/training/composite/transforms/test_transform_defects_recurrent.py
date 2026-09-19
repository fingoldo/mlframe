"""Regression tests for batch-state defects in the recurrent / positional transforms (EWMA, volatility, rolling ratio, frac-diff, seasonal and their
grouped variants).

The recurrent inverses restart their state at every batch boundary, so the same rows scored in one batch or in chunks got different predictions, and
the continuation seeds were means rather than the actual train tail. The seasonal transform picked the largest candidate period on noise and assumed
every predict batch starts at phase 0.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.composite.transforms import get_transform
from mlframe.utils.log_throttle import reset_throttle_counts


def _random_walk(n: int, seed: int) -> np.ndarray:
    """A positive random walk (keeps ratio-style transforms in their domain)."""
    rng = np.random.default_rng(seed)
    return 100.0 + np.cumsum(rng.normal(0, 1, n))


@pytest.mark.parametrize("name", ["ewma_residual", "volatility_normalized_residual", "rolling_quantile_ratio"])
def test_base_recurrent_chunked_inverse_with_history_equals_one_batch(name: str) -> None:
    """Scoring a continuation one row at a time differed from one batch by up to 11.94 (EWMA) because each call re-seeded the state."""
    base = _random_walk(400, 0)
    y = base + np.random.default_rng(1).normal(0, 0.5, 400)
    tr = get_transform(name)
    p = tr.fit(y[:300], base[:300])
    t_hat = tr.forward(y[300:], base[300:], p)
    full = tr.inverse(t_hat, base[300:], p)
    chunked = np.concatenate([tr.inverse(t_hat[i : i + 1], base[300 + i : 301 + i], p, history_base=base[300 : 300 + i]) for i in range(100)])
    np.testing.assert_array_equal(chunked, full)
    single = np.concatenate([tr.inverse(t_hat[i : i + 1], base[300 + i : 301 + i], p) for i in range(100)])
    assert float(np.max(np.abs(single - full))) > 1e-6


def test_frac_diff_observed_history_removes_bias_amplification() -> None:
    """Batch inverse feeds its own reconstructions back in, so a T-bias of delta became ~9.75*delta in y; observed history gives gain 1."""
    y = _random_walk(400, 2)
    tr = get_transform("frac_diff")
    p = tr.fit(y[:300], None)
    t_true = tr.forward(y, None, p)[300:]
    delta = 0.1
    batch = tr.inverse(t_true + delta, None, p, history_y=y[:300])
    assert float(batch[-1] - y[-1]) > 5 * delta
    online = np.concatenate([tr.inverse(t_true[i : i + 1] + delta, None, p, history_y=y[: 300 + i]) for i in range(100)])
    np.testing.assert_allclose(online - y[300:], delta, rtol=1e-9)


def test_frac_diff_continuation_uses_the_actual_train_tail() -> None:
    """The continuation seed padded every lag with the MEAN of the last lags values; a forward over train+continuation reads the actual values."""
    y = _random_walk(400, 3)
    tr = get_transform("frac_diff")
    p = dict(tr.fit(y[:300], None), recurrence_continuation=True)
    t_cont = tr.forward(y, None, p)[300:]
    np.testing.assert_allclose(tr.inverse(t_cont, None, p), y[300:], rtol=1e-12, atol=1e-9)


def test_rolling_quantile_ratio_continuation_reads_the_train_tail_window() -> None:
    """Under continuation the trailing window of the first predict rows must include the last train rows, not truncate at the batch start."""
    base = _random_walk(400, 4)
    y = base * 1.5
    tr = get_transform("rolling_quantile_ratio")
    p = dict(tr.fit(y[:300], base[:300]), recurrence_continuation=True)
    t_cont = tr.forward(y, base, p)[300:]
    np.testing.assert_allclose(tr.inverse(t_cont, base[300:], p), y[300:], rtol=1e-12)


def test_recurrent_inverse_warns_on_a_cold_short_batch(caplog: pytest.LogCaptureFixture) -> None:
    """A batch shorter than the recurrence horizon silently restarted from the train-mean seed; that must now be visible in the log."""
    base = _random_walk(200, 5)
    tr = get_transform("ewma_residual")
    p = tr.fit(base, base)
    reset_throttle_counts("recurrent_cold_start_ewma_residual")
    with caplog.at_level(logging.WARNING):
        tr.inverse(np.zeros(1), base[:1], p)
    assert any("restarts the recurrence" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("seed", range(10))
def test_seasonal_noise_selects_no_seasonality(seed: int) -> None:
    """In-sample variance never rises with the period, so pure noise selected period 52 in 20/20 seeds."""
    y = np.random.default_rng(seed).standard_normal(400)
    assert get_transform("seasonal_residual").fit(y, None)["period"] == 1


def test_seasonal_true_period_beats_its_multiple() -> None:
    """A clean period-12 series selected 24, which nests 12 and has twice as many phase means."""
    rng = np.random.default_rng(0)
    pattern = rng.normal(0, 3, 12)
    y = np.tile(pattern, 40) + rng.normal(0, 1, 480)
    assert get_transform("seasonal_residual").fit(y, None)["period"] == 12


def test_seasonal_continuation_starts_at_the_true_phase() -> None:
    """A batch continuing a 1000-row period-7 series starts at phase 1000 % 7 = 6; the inverse assumed phase 0 and shifted every prediction."""
    pattern = np.array([1.0, 5.0, -2.0, 0.0, 3.0, 7.0, -4.0])
    y = np.tile(pattern, 160)[:1070]
    tr = get_transform("seasonal_residual")
    p = dict(tr.fit(y[:1000], None, period=7), recurrence_continuation=True)
    np.testing.assert_allclose(tr.inverse(np.zeros(70), None, p), y[1000:], atol=1e-12)
    np.testing.assert_allclose(tr.inverse(np.zeros(70), None, p, row_index=np.arange(1000, 1070)), y[1000:], atol=1e-12)


def test_grouped_ewma_unseen_group_continuation_uses_the_ungrouped_tail_seed() -> None:
    """``tail_anchor`` of the grouped variant was the whole-history MEAN, not the train-tail state the ungrouped transform continues from."""
    base = _random_walk(300, 6)
    groups = np.repeat(np.array(["a", "b", "c"]), 100)
    grouped = get_transform("ewma_residual_grouped").fit(base, base, groups=groups)
    ungrouped = get_transform("ewma_residual").fit(base, base)
    assert grouped["tail_anchor"] == pytest.approx(ungrouped["tail_anchor"])
    p = dict(grouped, recurrence_continuation=True)
    out = get_transform("ewma_residual_grouped").inverse(np.zeros(1), np.array([50.0]), p, groups=np.array(["zz"]))
    alpha = 2.0 / (int(p["k"]) + 1.0)
    assert out[0] == pytest.approx((1 - alpha) * ungrouped["tail_anchor"] + alpha * 50.0)


def test_grouped_frac_diff_unseen_group_continuation_uses_the_ungrouped_tail() -> None:
    """An unseen group under continuation was seeded from the global mean instead of the ungrouped series' train tail."""
    y = _random_walk(300, 7)
    groups = np.repeat(np.array(["a", "b", "c"]), 100)
    grouped = dict(get_transform("frac_diff_grouped").fit(y, None, groups=groups), recurrence_continuation=True)
    ungrouped = dict(get_transform("frac_diff").fit(y, None), recurrence_continuation=True)
    np.testing.assert_allclose(grouped["tail_y"], ungrouped["tail_y"])
    t = np.array([0.3, -0.2])
    np.testing.assert_allclose(
        get_transform("frac_diff_grouped").inverse(t, None, grouped, groups=np.array(["zz", "zz"])),
        get_transform("frac_diff").inverse(t, None, ungrouped),
        rtol=1e-12,
    )


@pytest.mark.parametrize("name", ["ewma_residual_grouped", "rolling_quantile_ratio_grouped"])
def test_grouped_recurrent_chunked_inverse_with_history_equals_one_batch(name: str) -> None:
    """The grouped recurrences reset per call too; with each group's preceding rows as history, chunked scoring must match one batch."""
    base = np.abs(_random_walk(360, 8)) + 1.0
    groups = np.tile(np.array(["a", "b", "c"]), 120)
    y = base * 1.1
    tr = get_transform(name)
    p = tr.fit(y[:240], base[:240], groups=groups[:240])
    t_hat = tr.forward(y[240:], base[240:], p, groups=groups[240:])
    full = tr.inverse(t_hat, base[240:], p, groups=groups[240:])
    parts = []
    for lo in range(0, 120, 7):
        hi = min(lo + 7, 120)
        parts.append(
            tr.inverse(t_hat[lo:hi], base[240 + lo : 240 + hi], p, groups=groups[240 + lo : 240 + hi],
                       history_base=base[240 : 240 + lo], history_groups=groups[240 : 240 + lo])
        )
    np.testing.assert_array_equal(np.concatenate(parts), full)
