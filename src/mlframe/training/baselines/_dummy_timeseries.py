"""Timestamp / period / monotonicity helpers for ``dummy_baselines``.

Split out of ``dummy_baselines.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the moved helpers so the orchestrator continues to call them via the same
names.

What lives here:
  - ``_normalize_timestamps`` (Any -> int64 ns ndarray normalisation)
  - ``_is_temporally_monotonic`` (per-split monotonicity gate)
  - ``_infer_ts_step_periods`` (detect periodic structure from step
    spacings)
  - ``_detect_acf_periods`` (ACF-based period detection)
  - ``_resolve_ts_periods`` (per-target period resolver)
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_timestamps(ts: Any) -> np.ndarray | None:
    """Coerce timestamps to a 1-D numpy array."""
    if ts is None:
        return None
    try:
        if hasattr(ts, "to_numpy"):
            ts = ts.to_numpy()
        ts = np.asarray(ts)
        if ts.ndim != 1:
            ts = ts.ravel()
        # If datetime-like, convert to int64 nanoseconds for diff arithmetic.
        if ts.dtype.kind == "M":
            ts = ts.astype("datetime64[ns]").astype("int64")
        elif ts.dtype.kind == "O":
            # Object array: could be pd.Timestamp instances, datetime strings, OR
            # bare numeric ints that fell into an object Series (e.g. from a
            # df['ts'].values on a mixed-dtype frame). Sniff the first non-null
            # element: numeric ints go through the audit unit-detector to avoid
            # the ns-default trap; otherwise let pandas parse strings / Timestamps.
            _first_non_null = next(
                (v for v in ts if v is not None and not (isinstance(v, float) and np.isnan(v))),
                None,
            )
            if isinstance(_first_non_null, (int, np.integer)) and not isinstance(_first_non_null, bool):
                from ..targets import coerce_timestamps_for_audit as _coerce_ts
                _num = np.asarray([v if v is not None else 0 for v in ts], dtype=np.int64)
                # coerce returns datetime64[ns]; view-cast directly to int64 ns since epoch.
                ts = _coerce_ts(_num).view("int64")
            else:
                # tz_convert(None) strips tz so the .view("int64") below produces ns-since-epoch
                # rather than ns-since-tz-offset-epoch (which the prior .astype("datetime64[ns]")
                # could not do at all on tz-aware DatetimeIndex -- raised TypeError that the outer
                # try/except swallowed, returning None and silently disabling temporal logic).
                _dti = pd.to_datetime(ts, utc=True, errors="coerce")
                _n_nat = int(pd.isna(_dti).sum())
                if _n_nat > 0:
                    logger.warning(
                        "dummy_timeseries: %d timestamp value(s) were unparseable and coerced to NaT.", _n_nat,
                    )
                if getattr(_dti, "tz", None) is not None:
                    _dti = _dti.tz_convert(None)
                # Pin the datetime64 unit to ns BEFORE .view("int64"). pandas
                # 2.x can surface DatetimeIndex.to_numpy() at us / s units on
                # some inputs (e.g. fresh-build pandas / arrow combos) and
                # .view("int64") reads bytes as ints whose MEANING depends on
                # the underlying unit. Downstream callers (and the test
                # ::test_object_array_of_pd_timestamps_still_parsed) expect
                # ns-since-epoch; a unit mismatch silently rounds the data
                # into the 1970-epoch garbage band.
                ts = _dti.to_numpy(dtype="datetime64[ns]").view("int64")
        # Else assume already numeric (epoch ints, floats).
        return ts
    except (TypeError, ValueError, AttributeError, pd.errors.OutOfBoundsDatetime) as _exc:
        # Narrow swallow: only the failures we expect from malformed timestamp input
        # (unsupported dtype, unparseable strings, out-of-range datetime, missing
        # to_numpy attribute). Anything else (MemoryError, KeyboardInterrupt, our own
        # bugs) propagates so the operator notices. Pre-fix this caught bare Exception
        # and silently disabled every downstream temporal-monotonicity check with no
        # log line; the next call to _is_temporally_monotonic then defaulted to
        # "no temporal structure" without anyone knowing why.
        logger.warning(
            "_normalize_timestamps: ts coercion failed (%s: %s); temporal-monotonicity "
            "checks for this split will be skipped. Inspect the ts column dtype/values.",
            type(_exc).__name__, _exc,
        )
        return None


def _is_temporally_monotonic(
    ts_train: np.ndarray, ts_val: np.ndarray, ts_test: np.ndarray
) -> bool:
    """Strict monotonic split: train.max() <= val.min() AND val.max() <= test.min()."""
    if len(ts_train) == 0 or len(ts_val) == 0 or len(ts_test) == 0:
        return False
    return (
        ts_train.max() <= ts_val.min() and ts_val.max() <= ts_test.min()
    )


def _infer_ts_step_periods(ts_train: np.ndarray) -> tuple[str, list[int]]:
    """Step-size auto-infer; uses ``np.unique`` to handle duplicates.

    Returns ``(step_label, default_periods_for_that_step)``.
    """
    if len(ts_train) < 2:
        return "unknown", []
    unique_ts = np.unique(ts_train)
    if len(unique_ts) < 2:
        return "all-duplicate", []
    diffs = np.diff(unique_ts)
    if len(diffs) == 0:
        return "unknown", []
    median_diff = float(np.median(diffs))
    # Heuristic buckets (assuming int64 ns when datetime, or arbitrary float otherwise).
    NS_PER_HOUR = 3600 * 1e9
    NS_PER_DAY = 24 * NS_PER_HOUR
    NS_PER_WEEK = 7 * NS_PER_DAY
    NS_PER_MONTH = 30 * NS_PER_DAY
    if median_diff <= 0:
        return "duplicate-median", []
    if median_diff < 0.5 * NS_PER_HOUR:
        return "sub-hourly", [1]
    if median_diff < 1.5 * NS_PER_HOUR:
        return "hourly", [1, 24, 168]
    if median_diff < 1.5 * NS_PER_DAY:
        return "daily", [1, 7, 30, 365]
    if median_diff < 1.5 * NS_PER_WEEK:
        return "weekly", [1, 4, 52]
    if median_diff < 1.5 * NS_PER_MONTH:
        return "monthly", [1, 12]
    return "irregular", [1]


def _detect_acf_periods(y_train: np.ndarray, n_train: int) -> list[int]:
    """ACF-based period detection (differencing + stratified sample).

    Uses statsmodels.tsa.stattools.acf on first-differenced y_train.
    Returns top-2 peaks above 2/sqrt(n) threshold, filtered to
    ``2 <= P <= n_train // 4``.

    statsmodels imported lazily inside the function (D17): import
    failure -> empty list + INFO log, not module-load failure.
    """
    try:
        from statsmodels.tsa.stattools import acf
    except ImportError as e:
        logger.info(
            "[dummy-baselines] statsmodels unavailable (%s); ACF period "
            "detection skipped, using step-size defaults only", e,
        )
        return []

    # Stratified sample: for very large n_train, take a
    # uniform-random sample of contiguous-windowed sub-segments
    # (preserves local autocorrelation). Cap at 50000 rows for ACF.
    if n_train > 50_000:
        rng = np.random.default_rng(42)
        # Take 5 contiguous windows of 10000 rows each, randomly placed.
        window_size = 10_000
        n_windows = 5
        max_start = n_train - window_size
        starts = sorted(rng.integers(0, max_start, size=n_windows))
        sample_idx = np.concatenate([np.arange(s, s + window_size) for s in starts])
        y_sample = y_train[sample_idx]
    else:
        y_sample = y_train

    # First-differenced series: removes linear trend so
    # ACF peaks reflect seasonality, not trend.
    if len(y_sample) < 30:
        return []
    try:
        y_diff = np.diff(y_sample)
        nlags = min(int(10 * np.log10(len(y_diff))), len(y_diff) // 2)
        if nlags < 2:
            return []
        acf_vals = acf(y_diff, nlags=nlags, fft=True)
    except Exception as e:
        logger.info("[dummy-baselines] ACF computation failed (%s); skipping ACF peaks", e)
        return []

    # Significance threshold ~ 2/sqrt(n) (Bartlett 95% CI under white-noise null).
    threshold = 2.0 / np.sqrt(len(y_diff))
    peaks: list[tuple[int, float]] = []
    # Lag 0 always ACF=1; skip. Find local maxima above threshold.
    for lag in range(2, len(acf_vals)):
        v = acf_vals[lag]
        if abs(v) > threshold:
            # Local maximum check (avoid trend echo).
            is_peak = (lag == len(acf_vals) - 1 or v >= acf_vals[lag + 1]) and v >= acf_vals[lag - 1]
            if is_peak:
                peaks.append((lag, abs(v)))
    # Top-2 by absolute correlation, filtered by Nyquist-ish constraint.
    peaks.sort(key=lambda kv: -kv[1])
    candidate_periods: list[int] = []
    max_period = n_train // 4  # 4 cycles minimum
    for lag, _ in peaks[:5]:  # consider top-5, filter, take top-2 surviving
        if 2 <= lag <= max_period:
            candidate_periods.append(lag)
        if len(candidate_periods) >= 2:
            break
    return candidate_periods


def _resolve_ts_periods(
    y_train: np.ndarray,
    ts_train: np.ndarray,
    extra_periods: Sequence[int] = (),
) -> tuple[list[int], dict[str, Any]]:
    """Combine step-size + ACF + user-extra periods into final candidate list.

    Returns ``(periods, diagnostics_dict)``.
    """
    n_train = len(y_train)
    diagnostics: dict[str, Any] = {}

    # Step inference rejection gates.
    unique_ts = np.unique(ts_train)
    duplicate_threshold = max(10, int(0.01 * n_train))
    if len(unique_ts) < duplicate_threshold:
        diagnostics["rejected"] = (
            f"timestamps mostly-duplicate (unique={len(unique_ts)}/{n_train}); "
            "TS baselines disabled -- likely event-style data"
        )
        return [], diagnostics

    if n_train < 30:
        diagnostics["rejected"] = f"n_train={n_train} < 30 (ACF would be noise)"
        return [], diagnostics

    step_label, step_periods = _infer_ts_step_periods(ts_train)
    diagnostics["step_label"] = step_label
    diagnostics["step_periods"] = step_periods

    acf_periods = _detect_acf_periods(y_train, n_train)
    diagnostics["acf_periods"] = acf_periods

    # Combine: step + acf + user-extra, dedup, cap at 5, sort.
    combined: list[int] = []
    for p in list(step_periods) + list(acf_periods) + list(extra_periods):
        if isinstance(p, (int, np.integer)) and p >= 1 and p not in combined:
            combined.append(int(p))
    combined.sort()
    if len(combined) > 5:
        combined = combined[:5]
    diagnostics["using"] = combined
    return combined, diagnostics


# ---------------------------------------------------------------------
# Per-group baseline (cardinality cap + coverage + entity overlap)
# ---------------------------------------------------------------------
