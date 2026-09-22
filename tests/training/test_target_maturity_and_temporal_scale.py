"""Regression tests for the training-log audit 2026-09-20, sensor group.

- ``SEN-01`` the temporal audit compared an ABSOLUTE segment-rate spread against a RELATIVE threshold, so any
  regression target whose scale exceeded 0.1 was reported unstable (a production target with a 5% spread warned
  against a 10% threshold).
- ``SEN-03`` nothing detected a still-accruing (right-censored) target, the mechanism behind a production run's
  negative test R2.
- ``SEN-04`` the dropped-thin-bin note did not say which bins were dropped, so a verdict computed over a window
  that excluded the whole evaluation period read as an all-clear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.targets.target_maturity_audit import (
    audit_binned_target_maturity,
    audit_target_maturity,
)
from mlframe.training.targets.target_temporal_audit import audit_target_over_time

DAYS = 189
N = 60_000


def _timeline(seed: int = 0):
    """``(timestamps, days_since_start)`` for a uniformly-sampled daily timeline."""
    rng = np.random.default_rng(seed)
    t0 = np.datetime64("2026-03-01")
    ts = t0 + rng.integers(0, DAYS, N).astype("timedelta64[D]")
    days = (ts - t0).astype("timedelta64[D]").astype(int)
    return ts, days, rng


def _frame(ts, y):
    return pd.DataFrame({"ts": ts, "y": y})


# --------------------------------------------------------------------------------------------------
# SEN-01: absolute-vs-relative spread
# --------------------------------------------------------------------------------------------------


def test_temporal_audit_does_not_warn_on_small_relative_spread_at_large_scale():
    """A regression target whose segments differ by 5% must not be reported unstable against a 10% threshold.

    The production case: ``target_hours_to_hire`` segments 309.114 / 324.579 -- spread 15.465 in target units,
    5.0% relative. The pre-fix predicate compared 15.465 against 0.10 and always warned.
    """
    ts, days, rng = _timeline()
    y = np.where(days < 40, 309.114, 324.579) + rng.normal(0, 0.5, N)
    result = audit_target_over_time(_frame(ts, y), "ts", "y", granularity="week", target_type="regression")
    assert not any("NOT stable" in w for w in result.warnings), result.warnings


def test_temporal_audit_still_warns_on_large_relative_spread_at_large_scale():
    """The same rescaling must not mute a real move: 130.451 -> 40.984 is a 69% relative spread."""
    ts, days, rng = _timeline()
    y = np.where(days < 60, 130.451, 40.984) + rng.normal(0, 0.5, N)
    result = audit_target_over_time(_frame(ts, y), "ts", "y", granularity="week", target_type="regression")
    unstable = [w for w in result.warnings if "NOT stable" in w]
    assert unstable, result.warnings
    assert "relative spread" in unstable[0]


def test_temporal_audit_keeps_absolute_threshold_for_binary_targets():
    """A probability-valued rate keeps the percentage-point rule: a 0.20 swing must still warn at threshold 0.10."""
    ts, days, rng = _timeline()
    p = np.where(days < 90, 0.20, 0.40)
    y = (rng.random(N) < p).astype(int)
    result = audit_target_over_time(_frame(ts, y), "ts", "y", granularity="week", target_type="binary_classification")
    unstable = [w for w in result.warnings if "NOT stable" in w]
    assert unstable, result.warnings
    assert "rates are probabilities" in unstable[0]


# --------------------------------------------------------------------------------------------------
# SEN-03: right-censoring vs regime change
# --------------------------------------------------------------------------------------------------


def test_maturity_audit_flags_a_still_accruing_target():
    """A target that accrues from nothing and is cut off at extract time must be identified as censored."""
    ts, days, rng = _timeline()
    elapsed = (DAYS - days) / DAYS
    y = rng.gamma(2.0, 50.0, N) * elapsed
    result = audit_target_maturity(timestamps=ts, y=y, target_name="target_total_charge")
    assert result.verdict == "censoring_likely", (result.verdict, result.trend, result.origin_ratio)
    assert result.origin_ratio < 0.25
    assert any("UNFINISHED labels" in w for w in result.warnings)


def test_maturity_audit_does_not_call_a_declining_regime_censoring():
    """A level that falls but whose newest rows still realise 60% of the old level cannot be censoring.

    Censoring forces the statistic to ~0 at a zero observation window; a regime change does not, and the two are
    otherwise indistinguishable from the marginal alone. This is the assertion that keeps the sensor honest.
    """
    ts, days, rng = _timeline()
    y = rng.gamma(2.0, 50.0, N) * np.interp(days, [0, DAYS], [1.0, 0.6])
    result = audit_target_maturity(timestamps=ts, y=y, target_name="t")
    assert result.verdict == "declining_level", (result.verdict, result.origin_ratio)
    assert result.origin_ratio > 0.25


def test_maturity_audit_reports_stable_for_a_stationary_target():
    ts, _days, rng = _timeline()
    y = rng.gamma(2.0, 50.0, N)
    assert audit_target_maturity(timestamps=ts, y=y).verdict == "stable"


def test_maturity_audit_reports_stable_for_a_rising_target():
    """Censoring can only depress the newest rows, so a target rising toward the present is never censored."""
    ts, days, rng = _timeline()
    y = rng.gamma(2.0, 50.0, N) * np.interp(days, [0, DAYS], [0.4, 1.0])
    assert audit_target_maturity(timestamps=ts, y=y).verdict == "stable"


def test_maturity_audit_on_the_production_bin_profile():
    """The weekly means logged for ``target_total_charge`` (130.451 / 84.484 / 40.984) read as censoring."""
    result = audit_binned_target_maturity(bin_stats=[130.451] * 4 + [84.484] * 11 + [40.984] * 9, target_name="target_total_charge")
    assert result.verdict == "censoring_likely", (result.verdict, result.origin_ratio)


def test_maturity_audit_needs_enough_bins():
    result = audit_binned_target_maturity(bin_stats=[5.0, 4.0, 3.0])
    assert result.verdict == "insufficient_data"
    assert not np.isfinite(result.trend)


def test_maturity_audit_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="timestamps has"):
        audit_target_maturity(timestamps=np.arange(10), y=np.arange(9))


def test_temporal_audit_surfaces_censoring_and_caveats_its_recommendation():
    """The wired path: the temporal audit must emit the censoring warning AND withdraw the 'train on the most
    recent segment' advice, which under censoring points at the least mature rows available."""
    ts, days, rng = _timeline()
    y = rng.gamma(2.0, 50.0, N) * ((DAYS - days) / DAYS)
    result = audit_target_over_time(_frame(ts, y), "ts", "y", granularity="week", target_type="regression")
    assert result.actionable["maturity_verdict"] == "censoring_likely"
    assert any("UNFINISHED labels" in w for w in result.warnings)
    assert "CAUTION" in result.actionable["recommendation"]
    assert "LEAST mature" in result.actionable["recommendation"]


# --------------------------------------------------------------------------------------------------
# SEN-04: dropped bins must name themselves
# --------------------------------------------------------------------------------------------------


def test_dropped_bin_warning_names_the_bins_and_the_covered_window():
    """A thin trailing bin must be named, and the audit must state the window it actually covers.

    In production five trailing bins were dropped and the surviving window ended five weeks before the test split
    did, while the note said only "5 bin(s) dropped".
    """
    ts, days, rng = _timeline()
    # Thin out the newest fortnight so its bins fall under the 0.5x-median-n_obs filter.
    keep = (days < DAYS - 14) | (rng.random(N) < 0.05)
    y = rng.gamma(2.0, 50.0, N)
    result = audit_target_over_time(_frame(ts[keep], y[keep]), "ts", "y", granularity="week", target_type="regression")
    dropped = [w for w in result.warnings if "dropped from the audit" in w]
    assert dropped, result.warnings
    assert "the audit therefore covers" in dropped[0]
    assert "MOST RECENT" in dropped[0]
    assert result.actionable["n_trailing_bins_dropped"] >= 1
    assert result.actionable["audit_covers"] is not None
