"""The device rolling window statistic equals the numba linked-list walk for every statistic, for datetime (int64 ns) and numeric times, with interleaved entities and NaN values; the dispatcher demonstrably runs the device."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _rolling_window_gpu as g
from mlframe.feature_selection.filters import _temporal_agg_fe_rolling as tr


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")


def _data(n=20_000, entities=120, datetime_time=False, interleaved=True):
    """Rows over entities (ascending time inside an entity, optionally interleaved in row order), ties in time and ~3 percent NaN values."""
    rng = np.random.default_rng(5)
    gc = rng.integers(0, entities, n).astype(np.int64)
    if not interleaved:
        gc = np.sort(gc)
    t = np.zeros(n)
    for e in np.unique(gc):
        idx = np.flatnonzero(gc == e)
        t[idx] = np.sort(np.round(rng.uniform(0.0, 200.0, idx.size)))
    vals = rng.standard_normal(n) + 0.01 * gc
    vals[rng.random(n) < 0.03] = np.nan
    if datetime_time:
        times = (np.datetime64("2024-01-01", "ns") + (t * 86_400e9).astype("int64").astype("timedelta64[ns]")).astype("datetime64[ns]")
        return times, vals, gc
    return t, vals, gc


@pytest.mark.parametrize("stat", ["count", "mean", "std", "min", "max"])
@pytest.mark.parametrize("datetime_time", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_device_equals_host_walk(stat, datetime_time, interleaved):
    """count/min/max exactly, mean and std to 1e-12, NaNs in the same places."""
    _needs_gpu()
    times, vals, gc = _data(datetime_time=datetime_time, interleaved=interleaved)
    window = "7D" if datetime_time else "7"
    host = tr._rolling_stat_past_only(times, vals, gc, window, stat)
    if datetime_time:
        t_num = np.ascontiguousarray(times.astype("datetime64[ns]").view("int64"))
        td = np.int64(pd.Timedelta(window).value)
    else:
        t_num, td = np.ascontiguousarray(times), float(tr._numeric_window(window))
    dev = g.rolling_stat_past_only_gpu(t_num, vals, gc, td, tr._EXPANDING_STAT_CODE[stat])
    assert np.array_equal(np.isnan(host), np.isnan(dev))
    if stat in ("count", "min", "max"):
        np.testing.assert_array_equal(dev, host)
    else:
        np.testing.assert_allclose(dev, host, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_dispatcher_uses_the_device_when_chosen_and_the_result_is_unchanged(monkeypatch):
    """With the device forced the wrapper returns the host's values and the device launcher is called once; with it off the launcher is not called."""
    _needs_gpu()
    times, vals, gc = _data(interleaved=False)
    monkeypatch.setattr(g, "rolling_device_chosen", lambda n: False)
    want = tr._rolling_stat_past_only(times, vals, gc, "7", "std")
    monkeypatch.setattr(g, "rolling_device_chosen", lambda n: True)
    calls = {"n": 0}
    real = g.rolling_stat_past_only_gpu

    def counting(*a, **k):
        """The device launcher, counting calls so the test proves the device path ran."""
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(g, "rolling_stat_past_only_gpu", counting)
    got = tr._rolling_stat_past_only(times, vals, gc, "7", "std")
    assert calls["n"] == 1
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_empty_input_and_device_fault_fall_back_to_the_host(monkeypatch):
    """A device fault returns the host's answer (the dispatcher yields None) and an empty frame never reaches the device."""
    times, vals, gc = _data(n=500, entities=10)
    monkeypatch.setattr(g, "rolling_device_chosen", lambda n: True)
    want_monkey = tr._rolling_stat_past_only(times, vals, gc, "7", "mean")

    def boom(*a, **k):
        """A launcher that faults."""
        raise RuntimeError("device fault")

    monkeypatch.setattr(g, "rolling_stat_past_only_gpu", boom)
    got = tr._rolling_stat_past_only(times, vals, gc, "7", "mean")
    np.testing.assert_allclose(got, want_monkey, equal_nan=True)
    assert tr._rolling_stat_past_only(times[:0], vals[:0], gc[:0], "7", "mean").size == 0
