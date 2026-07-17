"""Sensor for compute_ml_perf_by_time: int64 epoch-seconds timestamps
must NOT collapse the entire dataset into a single bucket.

Fixed at the same commit wave as the audit timestamp-unit fix (9a1855e + 3f11d93).
Without the coerce_timestamps_for_audit threading, ``pd.to_datetime(int64_arr)``
defaulted to nanosecond interpretation, every row landed in the first ~2 seconds
after 1970-01-01, ``pd.Grouper(freq='D')`` saw all rows in a single day bucket,
and the per-time-bin metric collapsed into a single value -- silently. The output
looked legitimate ("here's the metric over time"), but the time axis was a lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.evaluation import compute_ml_perf_by_time


def test_int64_epoch_seconds_produces_multiple_time_bins():
    """A 7-day span of int64 epoch-seconds at 1k rows/day must produce ~7 bins,
    not 1. Without the fix, all rows would collapse into a single bucket at 1970-01-01.
    """
    n_per_day = 1000
    n_days = 7
    rng = np.random.default_rng(42)
    # 7 days of epoch-seconds timestamps starting 2023-11-14
    base = 1_700_000_000
    ts = np.concatenate([base + day * 86_400 + rng.integers(0, 86_400, n_per_day) for day in range(n_days)])
    y_true = rng.integers(0, 2, len(ts))
    y_pred = rng.random(len(ts))

    out = compute_ml_perf_by_time(y_true, y_pred, ts, freq="D", metric="roc_auc", min_samples=100)

    # Pre-fix: all 7000 rows collapsed to 1970-01-01 -> 1 bin.
    # Post-fix: ~7 bins for the 7-day span.
    assert len(out) >= 5, f"expected ~7 daily bins for 7-day span; got {len(out)} bin(s). Suggests int64 epoch-seconds collapsed to ns interpretation again."
    # All bins must fall in late-2023, not 1970.
    bin_years = sorted({pd.Timestamp(b).year for b in out.index})
    assert bin_years == [2023] or bin_years == [2023, 2024], f"bins must be in 2023 (or 2023/2024 if span crosses year), got years {bin_years}"


def test_datetime64_input_still_works():
    """Datetime64 input continues to round-trip correctly (no regression on the path
    that was already correct pre-fix)."""
    n = 5000
    ts = pd.date_range("2023-11-14", periods=n, freq="h").to_numpy()
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, n)
    y_pred = rng.random(n)
    out = compute_ml_perf_by_time(y_true, y_pred, ts, freq="D", metric="roc_auc", min_samples=10)
    # 5000 hours = ~208 days; expect ~208 bins.
    assert 150 < len(out) < 250, f"expected ~208 daily bins, got {len(out)}"


def test_int64_epoch_milliseconds_auto_detected():
    """ms-scale epoch input should also auto-detect (1.7e12 range)."""
    n_per_day = 500
    n_days = 5
    rng = np.random.default_rng(42)
    base = 1_700_000_000_000  # epoch ms
    ts = np.concatenate([base + day * 86_400_000 + rng.integers(0, 86_400_000, n_per_day) for day in range(n_days)])
    y_true = rng.integers(0, 2, len(ts))
    y_pred = rng.random(len(ts))
    out = compute_ml_perf_by_time(y_true, y_pred, ts, freq="D", metric="roc_auc", min_samples=50)
    assert len(out) >= 3, f"expected ~5 daily bins; got {len(out)}"


# ---------------------------------------------------------------------------
# PERF-14: the day-divisor numpy fast path must be byte-identical to pd.Grouper.
# ---------------------------------------------------------------------------


def _grouper_reference(y_true, y_pred, timestamps, freq, metric, min_samples):
    """Independent pd.Grouper reference for the fast-path output (mirrors the slow branch).

    Uses the same timestamp coercion + per-bin metric the function uses, so any difference is
    attributable to the numpy argsort + run-length-slice fast path, not coercion / metric drift.
    """
    from mlframe.training.evaluation import _compute_metric, _normalize_pandas_offset_alias
    from mlframe.training.targets import coerce_timestamps_for_audit

    ts = coerce_timestamps_for_audit(np.asarray(timestamps))
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred, dtype=float), "ts": ts})
    df = df.set_index("ts").sort_index()
    rows = []
    for bin_start, chunk in df.groupby(pd.Grouper(freq=_normalize_pandas_offset_alias(freq))):
        n = len(chunk)
        if n == 0:
            continue
        if n < min_samples:
            val = float("nan")
        else:
            val = _compute_metric(metric, chunk["y_true"].values, chunk["y_pred"].values)
        rows.append({"bin": bin_start, metric: val, "n_samples": n})
    return pd.DataFrame(rows).set_index("bin") if rows else pd.DataFrame(columns=[metric, "n_samples"])


def test_numpy_fast_path_byte_identical_to_grouper_on_day_divisor_offset():
    """6h bins (a day-divisor offset) take the numpy argsort fast path; its bins, metric values,
    and n_samples must EXACTLY match the pd.Grouper reference. The timestamps deliberately start
    off-midnight so the epoch-grid floor alignment is non-trivially exercised."""
    rng = np.random.default_rng(7)
    # Start at 2023-11-14 05:17:33 (off any 6h boundary) so floor() alignment is a real test.
    base = pd.Timestamp("2023-11-14 05:17:33").value // 1_000_000_000  # epoch seconds
    n_per_day = 800
    n_days = 6
    ts = np.concatenate([base + day * 86_400 + rng.integers(0, 86_400, n_per_day) for day in range(n_days)])
    y_true = rng.integers(0, 2, len(ts))
    y_pred = rng.random(len(ts))

    fast = compute_ml_perf_by_time(y_true, y_pred, ts, freq="6h", metric="roc_auc", min_samples=1)
    ref = _grouper_reference(y_true, y_pred, ts, freq="6h", metric="roc_auc", min_samples=1)

    # Same bins (labels + order), same n_samples, same metric values (exact).
    pd.testing.assert_index_equal(fast.index, ref.index)
    np.testing.assert_array_equal(fast["n_samples"].to_numpy(), ref["n_samples"].to_numpy())
    np.testing.assert_array_equal(fast["roc_auc"].to_numpy(), ref["roc_auc"].to_numpy())


def test_numpy_fast_path_matches_grouper_for_hourly_mse():
    """Hourly bins (a day-divisor offset), mse metric. Bins + n_samples are byte-identical.

    The per-bin mse itself matches the Grouper reference only to floating-point ULP (~1e-15),
    NOT bit-exactly: the fast path's argsort is non-stable, so within a bin the row ORDER differs
    from Grouper's stable sort, and a sum-of-squares reduction is order-dependent at the last ULP.
    That divergence is far below any selection / reporting threshold, and it does not occur for the
    rank-based metrics (roc_auc / average_precision) the docstring's byte-identity claim targets."""
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2024-01-02 23:41:07").value // 1_000_000_000
    n = 6000
    ts = base + rng.integers(0, 5 * 86_400, n)
    y_true = rng.random(n)
    y_pred = y_true + rng.normal(scale=0.2, size=n)

    fast = compute_ml_perf_by_time(y_true, y_pred, ts, freq="h", metric="mse", min_samples=1)
    ref = _grouper_reference(y_true, y_pred, ts, freq="h", metric="mse", min_samples=1)

    pd.testing.assert_index_equal(fast.index, ref.index)
    np.testing.assert_array_equal(fast["n_samples"].to_numpy(), ref["n_samples"].to_numpy())
    np.testing.assert_allclose(fast["mse"].to_numpy(), ref["mse"].to_numpy(), rtol=1e-12, atol=1e-15)


def test_numpy_fast_path_byte_identical_for_average_precision():
    """A second rank-based metric (average_precision) must be EXACTLY byte-identical to Grouper:
    rank metrics are order-invariant so the non-stable argsort cannot perturb the value."""
    rng = np.random.default_rng(13)
    base = pd.Timestamp("2023-07-01 11:23:45").value // 1_000_000_000
    n_per_day = 600
    n_days = 5
    ts = np.concatenate([base + day * 86_400 + rng.integers(0, 86_400, n_per_day) for day in range(n_days)])
    y_true = rng.integers(0, 2, len(ts))
    y_pred = rng.random(len(ts))

    fast = compute_ml_perf_by_time(y_true, y_pred, ts, freq="12h", metric="average_precision", min_samples=1)
    ref = _grouper_reference(y_true, y_pred, ts, freq="12h", metric="average_precision", min_samples=1)

    pd.testing.assert_index_equal(fast.index, ref.index)
    np.testing.assert_array_equal(fast["n_samples"].to_numpy(), ref["n_samples"].to_numpy())
    np.testing.assert_array_equal(fast["average_precision"].to_numpy(), ref["average_precision"].to_numpy())
