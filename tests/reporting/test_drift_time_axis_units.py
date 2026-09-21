"""Time-axis charts must place polars' microsecond datetimes at their real dates.

A production test split from 2026-08-10 to 2026-09-13 was drawn as 1970-01-21 16:14..16:47 on the residual-vs-time and
CUSUM charts: ``datetime64[us]`` cast to float gives microseconds, which the renderers read as nanoseconds.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.drift import cusum_residual_drift, residual_vs_time


def _xs(spec) -> np.ndarray:
    xs = []
    for row in spec.panels:
        for panel in row:
            for attr in ("x", "xs"):
                v = getattr(panel, attr, None)
                if v is not None:
                    xs.append(np.asarray(v, dtype=np.float64).ravel())
    assert xs, "no x series found on the figure spec"
    return np.concatenate(xs)


def _data(unit: str):
    rng = np.random.default_rng(0)
    n = 2000
    ts = (np.datetime64("2026-08-10") + np.sort(rng.integers(0, 34 * 86400, n)).astype("timedelta64[s]")).astype(f"datetime64[{unit}]")
    y = rng.normal(size=n)
    return y, y + rng.normal(scale=0.1, size=n), ts


def _assert_2026(xs: np.ndarray):
    lo, hi = np.datetime64("2026-08-01").astype("datetime64[ns]").astype(np.int64), np.datetime64("2026-09-20").astype("datetime64[ns]").astype(np.int64)
    finite = xs[np.isfinite(xs)]
    assert finite.size and finite.min() >= lo and finite.max() <= hi


def test_residual_vs_time_microsecond_timestamps():
    for unit in ("us", "ns", "ms"):
        _assert_2026(_xs(residual_vs_time(*_data(unit))))


def test_cusum_microsecond_timestamps():
    y, p, ts = _data("us")
    spec = cusum_residual_drift(y, p, timestamps=ts)
    xs = _xs(spec)
    big = xs[np.abs(xs) > 1e15]
    _assert_2026(big)
