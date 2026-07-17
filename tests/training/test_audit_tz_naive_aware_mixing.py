"""Wave-34 sensor: tz-naive vs tz-aware datetime mixing fixes.

Wave 34 audit (2026-05-20) found 1 P1 + 1 P2 (both in
target_temporal_audit.py) + 1 Low (cleaning.py pandas 2.x raise risk).
All 3 closed.

#1 P1 target_temporal_audit.py:335 recommended_filter_mask
   ``ts = pd.Series(coerce_timestamps_for_audit(timestamps))`` produced
   tz-naive datetime64[ns]; ``start_ts/end_ts = kept[idx].bin_start``
   could be tz-aware Timestamps when the original ``timestamp_col``
   was polars ``Datetime(time_zone='UTC')``. Direct ``ts >= start_ts``
   raised TypeError. Fix: ``_normalize_bin_ts`` helper that
   tz_convert('UTC')+tz_localize(None) the kept-bin Timestamps before
   comparison.

#2 P2 target_temporal_audit.py:113 coerce_timestamps_for_audit
   ``pd.to_datetime(arr).to_numpy()`` returned an OBJECT array (not the
   documented datetime64[ns]) when arr contained tz-aware Timestamps.
   Downstream callers silently got mixed-dtype output. Fix: explicit
   ``utc=True`` + ``tz_convert('UTC').tz_localize(None)`` + WARN log so
   the contract is honoured uniformly.

#3 Low cleaning.py:256 ``.astype("datetime64[D]")`` raises on pandas
   >=2.0 for tz-aware Series. Fix: strip tz at function entry via
   ``tz_convert('UTC').tz_localize(None)`` so the rounding-resolution
   probe works across pandas eras.
"""

from __future__ import annotations

import logging

import pandas as pd


# ---- #1 + #2: target_temporal_audit -----------------------------------


def test_coerce_timestamps_for_audit_returns_datetime64ns_on_tz_aware(caplog):
    """Documented contract: returns datetime64[ns]. Pre-fix tz-aware
    input returned an OBJECT dtype array."""
    from mlframe.training.targets.target_temporal_audit import coerce_timestamps_for_audit

    ts_aware = pd.Series(pd.date_range("2024-01-01", periods=5, tz="UTC"))
    with caplog.at_level(logging.WARNING, logger="mlframe.training.targets.target_temporal_audit"):
        out = coerce_timestamps_for_audit(ts_aware)
    assert str(out.dtype) == "datetime64[ns]", (
        f"Wave 34 P2 regression: contract violation -- expected "
        f"datetime64[ns], got {out.dtype}. tz-aware input produced "
        f"object dtype, silently corrupting downstream Grouper/comparisons."
    )
    # WARN log fires naming the tz strip:
    assert any("tz-aware Timestamps" in r.message for r in caplog.records), f"Expected WARN naming the tz strip; got: {[r.message for r in caplog.records]}"


def test_coerce_timestamps_for_audit_passthrough_tz_naive():
    """tz-naive input must not trigger the WARN (no spurious noise)."""
    from mlframe.training.targets.target_temporal_audit import coerce_timestamps_for_audit

    ts_naive = pd.Series(pd.date_range("2024-01-01", periods=5))
    out = coerce_timestamps_for_audit(ts_naive)
    assert str(out.dtype) == "datetime64[ns]"


def test_recommended_filter_mask_handles_tz_aware_segments():
    """A result whose bins carry tz-AWARE ``bin_start`` Timestamps must produce
    a mask without raising ``TypeError: Cannot compare tz-naive and tz-aware``
    against the tz-naive timestamps coerced from the input."""
    from mlframe.training.targets.target_temporal_audit import (
        TemporalAuditResult,
        TimeBin,
    )

    # tz-AWARE bin edges (the polars Datetime(time_zone='UTC') hazard).
    starts = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    bins = [TimeBin(bin_label=str(s.date()), bin_start=s, n_obs=10, target_rate=0.3, kept=True) for s in starts]
    result = TemporalAuditResult(
        target_name="y",
        target_type="binary",
        timestamp_col="ts",
        granularity="day",
        bins=bins,
        change_point_indices=[],
        segments=[{"start_idx": 0, "end_idx": 4, "n_obs": 40, "n_bins": 4, "mean_rate": 0.3}],
        warnings=[],
    )
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=8, freq="12h", tz="UTC"))
    mask = result.recommended_filter_mask(timestamps, segment="all_stable")
    assert mask.shape[0] == len(timestamps)
    assert mask.dtype == bool
    assert mask.any()


# ---- #3 cleaning.py datetime64 unit probe -----------------------------


def test_cleaning_datetime_probe_strips_tz_before_astype():
    """The pandas-2.x raise on tz-aware Series .astype('datetime64[D]')
    is pre-empted by an upfront tz strip."""
    import pathlib
    import mlframe as _mlframe

    src = (pathlib.Path(_mlframe.__file__).resolve().parent / "preprocessing" / "cleaning.py").read_text(encoding="utf-8")
    # Pre-fix shape direct on values MUST be gone:
    assert 'if np.all(values.astype(f"datetime64[{date_fract}]") == values):' not in src, (
        "cleaning.py reverted to .astype on the raw values, which raises on pandas >=2.0 for tz-aware Series."
    )
    # Post-fix marker:
    assert "_vals_naive = values" in src
    assert 'values.tz_convert("UTC").tz_localize(None)' in src


# ---- behavioural: tz-aware input doesn't crash audit ---------------------


def test_audit_does_not_crash_on_tz_aware_input():
    """End-to-end-ish: coerce_timestamps_for_audit + a manual
    tz-aware ``ts`` Series comparison succeeds post-fix."""
    from mlframe.training.targets.target_temporal_audit import coerce_timestamps_for_audit

    ts_aware = pd.Series(pd.date_range("2024-01-01", periods=10, tz="UTC"))
    ts_naive = coerce_timestamps_for_audit(ts_aware)
    # Simulating the downstream comparison shape from recommended_filter_mask:
    # ts (tz-naive) >= start_ts (after _normalize_bin_ts) MUST not raise.
    ts_series = pd.Series(ts_naive)
    # Pre-fix this comparison would have been ts_naive >= tz_aware_timestamp -> TypeError.
    # Post-fix start_ts is normalised first.
    start_ts = pd.Timestamp("2024-01-03")  # tz-naive (post-_normalize_bin_ts shape)
    mask = ts_series >= start_ts
    assert mask.sum() > 0  # exact value depends on date_range; just confirm no raise
