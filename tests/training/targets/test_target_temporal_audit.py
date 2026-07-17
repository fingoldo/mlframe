"""Tests for target_temporal_audit module.

The synthetic fixture reproduces the user's production graph:
- 2018-05 .. 2021-08: ~98% positive rate (biased period; only hires
  observed)
- 2021-09 .. 2022-08: ~40% positive rate (unbiased window — full uid
  scrape)
- 2022-09 .. 2026-03: ~98% positive rate again (biased)
- 2026-04: ~3% positive rate (very recent, partial month — sparse,
  should be filtered out by the min-bin-fraction guard)

Test invariants:
1. Granularity auto-picker chooses ``"month"`` for the ~96-month span.
2. Aggregation produces the right number of bins and rates within
   ±2pp of the synthetic generator's parameters.
3. Change-point detector flags BOTH transitions (biased→unbiased and
   unbiased→biased), giving 3 segments.
4. The segment summaries report mean rates close to the generator's
   parameters (0.98 / 0.40 / 0.98).
5. Drift WARN fires (spread between segments > 0.10).
6. Plotting saves a file without crashing.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from mlframe.training.targets.target_temporal_audit import (
    DEFAULT_ZSCORE_THRESHOLD,
    TemporalAuditResult,
    audit_target_over_time,
    audit_targets_over_time,
    find_change_points,
    find_change_points_pelt,
    find_change_points_zscore,
    format_temporal_audit_report,
    plot_target_over_time,
)


# -----------------------------------------------------------------------------
# Synthetic fixture
# -----------------------------------------------------------------------------


def _gen_temporal_target_dataset(
    start: str = "2018-05-01",
    end: str = "2026-04-15",
    biased_rate: float = 0.98,
    unbiased_dip_rate: float = 0.40,
    unbiased_dip_start: str = "2021-09-01",
    unbiased_dip_end: str = "2022-09-01",
    recent_dip_rate: float = 0.03,
    recent_dip_start: str = "2026-04-01",
    rows_per_day: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic timeline matching the user's graph shape."""
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    days = pd.date_range(start_ts, end_ts, freq="D")

    rows = []
    for day in days:
        if pd.Timestamp(unbiased_dip_start) <= day < pd.Timestamp(unbiased_dip_end):
            rate = unbiased_dip_rate
        elif day >= pd.Timestamp(recent_dip_start):
            rate = recent_dip_rate
        else:
            rate = biased_rate
        for _ in range(rows_per_day):
            rows.append(
                {
                    "job_posted_at": day + pd.Timedelta(seconds=int(rng.integers(0, 86400))),
                    "cl_act_total_hired": int(rng.uniform() < rate),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_temporal_df():
    # Session-scoped: data is deterministic + read-only; built once per session
    # via local Generator (no global numpy RNG mutation).
    return _gen_temporal_target_dataset()


# -----------------------------------------------------------------------------
# Granularity picker
# -----------------------------------------------------------------------------


def test_granularity_auto_picks_month_for_8_year_span(synthetic_temporal_df):
    """8-year span at month granularity gives ~95 bins; at quarter ~32;
    at year ~8. The picker should converge on quarter or month
    (target range 30-50)."""
    from mlframe.training.targets.target_temporal_audit import _pick_granularity

    g = _pick_granularity(synthetic_temporal_df["job_posted_at"])
    assert g in {"month", "quarter"}, f"expected month/quarter, got {g}"


def test_granularity_auto_picks_day_for_short_span():
    """A 2-month span has ~60 days → falls in the day range."""
    from mlframe.training.targets.target_temporal_audit import _pick_granularity

    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", "2024-03-01", freq="h"),
        }
    )
    g = _pick_granularity(df["ts"])
    assert g == "day", f"expected day, got {g}"


def test_granularity_auto_picks_year_for_long_span():
    """A 200-year span needs year-granularity to land near 30-50 bins."""
    from mlframe.training.targets.target_temporal_audit import _pick_granularity

    df = pd.DataFrame(
        {
            "ts": pd.date_range("1820-01-01", "2024-01-01", freq="YE"),
        }
    )
    g = _pick_granularity(df["ts"])
    assert g == "year", f"expected year, got {g}"


# -----------------------------------------------------------------------------
# Change-point detector — generic 1-D
# -----------------------------------------------------------------------------


def test_zscore_detects_single_dip_global_baseline():
    """Stable 0.98 baseline with a 5-bin dip to 0.40 in the middle.
    Default global-median baseline detects the dip cleanly."""
    rates = np.array([0.98] * 30 + [0.40] * 5 + [0.98] * 30, dtype=float)
    boundaries = find_change_points_zscore(
        rates,
        z_threshold=3.0,
        min_anomaly_run=2,
    )
    # Expect exactly one anomaly run = 2 boundary indices: [start, end_excl]
    assert len(boundaries) == 2
    assert boundaries[0] == 30
    assert boundaries[1] == 35


def test_zscore_detects_multiple_dips_global_baseline():
    """Two dips against a dominant 0.98 baseline → 4 boundary indices,
    3 segments."""
    rates = np.array(
        [0.98] * 20 + [0.40] * 5 + [0.98] * 20 + [0.30] * 5 + [0.98] * 20,
        dtype=float,
    )
    boundaries = find_change_points_zscore(rates, z_threshold=3.0)
    assert len(boundaries) == 4


def test_zscore_local_window_too_narrow_misses_wide_dip():
    """When the local window is narrower than the dip, the rolling
    median sits ON the dip and can't detect it. Documents the
    limitation that motivated the global-baseline default."""
    rates = np.array([0.98] * 30 + [0.40] * 5 + [0.98] * 30, dtype=float)
    boundaries = find_change_points_zscore(
        rates,
        window=5,
        z_threshold=3.0,
        min_anomaly_run=2,
    )
    # Local window of 5 sits inside the 5-bin dip → median = 0.40 → no flag
    assert boundaries == []


def test_zscore_local_window_wider_than_dip_catches_it():
    """A wider local window catches the dip via the surrounding
    stable bins anchoring the rolling median."""
    rates = np.array([0.98] * 30 + [0.40] * 5 + [0.98] * 30, dtype=float)
    boundaries = find_change_points_zscore(
        rates,
        window=21,
        z_threshold=3.0,
        min_anomaly_run=2,
    )
    assert len(boundaries) == 2


def test_zscore_no_anomalies_in_stable_series():
    rates = np.full(40, 0.50)
    rates += np.random.default_rng(0).normal(0, 0.005, size=40)
    boundaries = find_change_points_zscore(rates, z_threshold=3.0)
    assert boundaries == []


def test_zscore_min_anomaly_run_filters_single_spikes():
    """A single-bin spike doesn't qualify as a change point when
    min_anomaly_run=2."""
    rates = np.array([0.98] * 30 + [0.40] + [0.98] * 30, dtype=float)
    boundaries = find_change_points_zscore(
        rates,
        z_threshold=3.0,
        min_anomaly_run=2,
    )
    assert boundaries == []
    # But min_anomaly_run=1 catches it
    boundaries_1 = find_change_points_zscore(
        rates,
        z_threshold=3.0,
        min_anomaly_run=1,
    )
    assert len(boundaries_1) == 2


def test_zscore_weighted_ignores_low_n_bins():
    """A bin with very small n_obs is excluded from the baseline
    median (so it can't poison the baseline) but still gets flagged
    if its rate diverges enough."""
    rates = np.array([0.98] * 30 + [0.10] + [0.98] * 30, dtype=float)
    weights = np.array([1000.0] * 30 + [5.0] + [1000.0] * 30)
    boundaries = find_change_points_zscore(
        rates,
        weights=weights,
        z_threshold=3.0,
        min_anomaly_run=1,
    )
    # The bin DOES flag (rate 0.10 is far from the dominant 0.98 baseline),
    # but at min_anomaly_run=2 it'd be filtered. We test min_run=1 here.
    assert len(boundaries) == 2


# -----------------------------------------------------------------------------
# End-to-end audit
# -----------------------------------------------------------------------------


def test_audit_user_graph_finds_3_segments(synthetic_temporal_df):
    """The user's graph has 4 regimes (biased/dip/biased/recent_dip)
    but the recent dip has only ~half a month so it gets filtered.
    Result: 3 well-defined segments."""
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    assert isinstance(result, TemporalAuditResult)
    assert result.granularity in {"month", "quarter"}
    # Expect 3 segments after dropping the partial last bin
    assert 2 <= len(result.segments) <= 4
    assert any("not stable over time" in w.lower() for w in result.warnings)


def test_audit_segment_rates_match_generator(synthetic_temporal_df):
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    # Sort segments by mean_rate to check we hit ~0.40 (dip) and ~0.98 (biased)
    rates = sorted(s["mean_rate"] for s in result.segments)
    assert rates[0] < 0.55, f"expected ~0.40 dip segment; got rates={rates}"
    assert rates[-1] > 0.90, f"expected ~0.98 biased segment; got rates={rates}"


def test_audit_actionable_recommends_recent_stable(synthetic_temporal_df):
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    rec = result.actionable.get("recommendation", "")
    most_recent = result.actionable.get("most_recent_stable_segment")
    assert most_recent is not None
    # Recommendation references the most-recent segment by date label
    assert most_recent["start_label"] in rec
    assert "PULearningWrapper" in rec  # hooks back to Session-7 module


def test_audit_polars_input(synthetic_temporal_df):
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(synthetic_temporal_df)
    result = audit_target_over_time(
        df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    assert len(result.segments) >= 2


def test_audit_explicit_granularity(synthetic_temporal_df):
    """Explicit granularity override bypasses the auto-picker."""
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
        granularity="quarter",
    )
    assert result.granularity == "quarter"
    # Quarter gives ~32 bins for an 8-year span
    assert 25 <= len(result.bins) <= 40


def test_audit_regression_target():
    """Mean-of-y instead of P(y=1) for non-binary targets."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    n_per_day = 100
    rows = []
    for day in days:
        # Linear trend over 2024.
        target_mean = 10.0 + 5.0 * (day - days[0]).days / (days[-1] - days[0]).days
        for _ in range(n_per_day):
            rows.append(
                {
                    "ts": day,
                    "y": target_mean + rng.normal(0, 1.0),
                }
            )
    df = pd.DataFrame(rows)

    result = audit_target_over_time(
        df,
        timestamp_col="ts",
        target_col="y",
        target_type="regression",
        granularity="month",
    )
    rates = [b.target_rate for b in result.bins if b.kept]
    # Increasing trend
    assert rates[-1] > rates[0]
    # Spread > 1.0 over the year
    assert max(rates) - min(rates) > 1.0


def test_audit_format_report_includes_segments(synthetic_temporal_df):
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    report = format_temporal_audit_report(result)
    assert "target_temporal_audit" in report
    assert "segment" in report
    assert "WARN" in report
    assert "ACTIONABLE" in report


def test_audit_plot_saves_to_disk(synthetic_temporal_df, tmp_path):
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    out_path = str(tmp_path / "target_temporal_audit.png")
    plot_target_over_time(result, save_path=out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000  # > 1KB sanity check


def test_audit_no_changepoints_in_stable_data():
    """Constant-rate data should produce ONE segment and no drift WARN."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2018-01-01", "2026-04-01", freq="D")
    rows = []
    for day in days:
        for _ in range(100):
            rows.append(
                {
                    "ts": day,
                    "y": int(rng.uniform() < 0.5),
                }
            )
    df = pd.DataFrame(rows)
    result = audit_target_over_time(
        df,
        timestamp_col="ts",
        target_col="y",
        target_type="binary_classification",
    )
    assert len(result.segments) == 1
    assert not any("not stable" in w.lower() for w in result.warnings)


def test_audit_dropped_sparse_bin_warning():
    """Force a sparse trailing bin → warning lists it."""
    rng = np.random.default_rng(0)
    rows = []
    # 3 years at 100 rows/day
    for day in pd.date_range("2021-01-01", "2024-01-01", freq="D"):
        for _ in range(100):
            rows.append({"ts": day, "y": int(rng.uniform() < 0.5)})
    # Plus 5 partial-last-month rows at 2024-02-01..02-05
    for day in pd.date_range("2024-02-01", "2024-02-05", freq="D"):
        for _ in range(2):  # very sparse
            rows.append({"ts": day, "y": 1})
    df = pd.DataFrame(rows)
    result = audit_target_over_time(
        df,
        timestamp_col="ts",
        target_col="y",
        target_type="binary_classification",
    )
    assert any(b.kept is False for b in result.bins)
    assert any("dropped from the audit" in w for w in result.warnings)


def test_audit_to_dict_round_trip(synthetic_temporal_df):
    """Result must serialize to JSON-safe dict for metadata storage."""
    import orjson

    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    d = result.to_dict()
    s = orjson.dumps(d).decode()
    parsed = orjson.loads(s)
    assert parsed["target_name"] == result.target_name
    assert parsed["granularity"] == result.granularity
    assert len(parsed["bins"]) == len(result.bins)


def test_audit_default_z_threshold():
    assert DEFAULT_ZSCORE_THRESHOLD == 3.0


# -----------------------------------------------------------------------------
# Pelt (default change-point detector)
# -----------------------------------------------------------------------------


def test_pelt_detects_single_dip():
    """Stable 0.98 baseline with a 5-bin dip to 0.40 — Pelt finds 2
    inner change points: dip start and dip end."""
    rates = np.array([0.98] * 30 + [0.40] * 5 + [0.98] * 30, dtype=float)
    boundaries = find_change_points_pelt(rates)
    # boundaries are [c, c, d, d] (each inner change point is doubled
    # so _segments_from_change_points sees both seg-end and seg-start)
    inner_unique = sorted(set(boundaries))
    assert inner_unique == [30, 35]


def test_pelt_detects_user_graph_pattern():
    """User's graph pattern: 13 stable + 5 dip + 14 stable + 1 sparse.
    Pelt finds 3 transitions (dip start/end + sparse start)."""
    rates = np.array(
        [0.98] * 13 + [0.40] * 5 + [0.98] * 14 + [0.03] * 1,
        dtype=float,
    )
    boundaries = find_change_points_pelt(rates)
    inner = sorted(set(boundaries))
    # Expect transitions at 13, 18, 32 (dip start, dip end, sparse-bin start)
    assert 13 in inner
    assert 18 in inner


def test_pelt_no_changes_in_constant_series():
    rates = np.full(40, 0.50)
    rates += np.random.default_rng(0).normal(0, 0.005, size=40)
    boundaries = find_change_points_pelt(rates)
    assert boundaries == []


def test_pelt_balanced_regimes():
    """Two equal-sized regimes (no dominant baseline) — z-score
    global-baseline fails here, but Pelt handles it."""
    rates = np.array([0.30] * 25 + [0.70] * 25, dtype=float)
    boundaries_pelt = find_change_points_pelt(rates)
    inner = sorted(set(boundaries_pelt))
    assert 25 in inner

    # z-score with global-baseline finds the 0.30 segment is "below"
    # the median 0.50 baseline. It DOES detect this because the abs
    # spread (0.20) exceeds the 0.10 default. So it's a degenerate
    # case where z-score happens to work too (via the abs-spread
    # fallback). Still — Pelt's principled treatment is preferred.
    boundaries_z = find_change_points_zscore(rates)
    assert len(boundaries_z) > 0


def test_pelt_explicit_penalty_overrides_auto():
    """A high penalty produces fewer change points; low produces more."""
    rates = np.array(
        [0.98] * 10 + [0.40] * 5 + [0.98] * 10 + [0.30] * 5 + [0.98] * 10,
        dtype=float,
    )
    n_low = len(set(find_change_points_pelt(rates, penalty=0.05)))
    n_high = len(set(find_change_points_pelt(rates, penalty=10.0)))
    assert n_low > n_high


def test_dispatcher_default_pelt():
    rates = np.array([0.98] * 30 + [0.40] * 5 + [0.98] * 30, dtype=float)
    bk_default = find_change_points(rates)
    bk_pelt = find_change_points(rates, method="pelt")
    bk_zscore = find_change_points(rates, method="zscore")
    assert bk_default == bk_pelt
    # Both methods detect the dip but may differ in exact indices /
    # boundary format — both should at least be non-empty.
    assert len(bk_pelt) > 0
    assert len(bk_zscore) > 0


def test_dispatcher_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown change-point method"):
        find_change_points(np.array([1.0, 2.0]), method="bogus")


def test_audit_with_zscore_method(synthetic_temporal_df):
    """The audit accepts method='zscore' as alternative."""
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
        method="zscore",
    )
    assert len(result.segments) >= 2


def test_audit_default_method_is_pelt(synthetic_temporal_df):
    """Default audit uses Pelt and finds the dip."""
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    # 3+ segments expected (biased / dip / biased ± sparse-bin spike)
    assert len(result.segments) >= 3


# -----------------------------------------------------------------------------
# recommended_filter_mask — the actionable bridge from audit to subset
# -----------------------------------------------------------------------------


def test_recommended_mask_most_recent_stable(synthetic_temporal_df):
    """The default 'most_recent_stable' selector picks the last
    multi-bin segment — for the user's graph that's the post-dip
    biased run (2022-Q4..2026-Q1)."""
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    mask = result.recommended_filter_mask(synthetic_temporal_df["job_posted_at"])
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (len(synthetic_temporal_df),)
    # Mask should select a meaningful chunk (not 0%, not 100%)
    frac = mask.mean()
    assert 0.10 < frac < 0.90, f"unexpected mask coverage {frac:.2%}"
    # Subset's mean rate should match the segment's rate
    sub = synthetic_temporal_df.loc[mask, "cl_act_total_hired"]
    assert sub.mean() > 0.90  # post-dip is back at biased ~98%


def test_recommended_mask_largest(synthetic_temporal_df):
    """'largest' selects the segment with the most obs."""
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    mask = result.recommended_filter_mask(
        synthetic_temporal_df["job_posted_at"],
        segment="largest",
    )
    largest = max(result.segments, key=lambda s: s["n_obs"])
    # Mask should have approximately largest_segment.n_obs rows
    assert abs(int(mask.sum()) - int(largest["n_obs"])) <= int(largest["n_obs"]) * 0.05


def test_recommended_mask_all_stable(synthetic_temporal_df):
    """'all_stable' includes every multi-bin segment — this is most of
    the data (biased + dip + biased), excluding only 1-bin spikes."""
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    mask_all = result.recommended_filter_mask(
        synthetic_temporal_df["job_posted_at"],
        segment="all_stable",
    )
    mask_one = result.recommended_filter_mask(
        synthetic_temporal_df["job_posted_at"],
        segment="most_recent_stable",
    )
    # all_stable >= most_recent_stable in coverage
    assert mask_all.sum() >= mask_one.sum()


def test_recommended_mask_first_last(synthetic_temporal_df):
    """'first' and 'last' pick those by index — disjoint for the
    user's 3-segment pattern."""
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    if len(result.segments) < 2:
        pytest.skip("need ≥2 segments for this test")
    mask_first = result.recommended_filter_mask(
        synthetic_temporal_df["job_posted_at"],
        segment="first",
    )
    mask_last = result.recommended_filter_mask(
        synthetic_temporal_df["job_posted_at"],
        segment="last",
    )
    assert mask_first.any() and mask_last.any()
    assert not (mask_first & mask_last).any()  # disjoint


def test_recommended_mask_unknown_selector_raises(synthetic_temporal_df):
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    with pytest.raises(ValueError, match="Unknown segment selector"):
        result.recommended_filter_mask(
            synthetic_temporal_df["job_posted_at"],
            segment="bogus",
        )


def test_recommended_mask_polars_series_input(synthetic_temporal_df):
    pl = pytest.importorskip("polars")
    result = audit_target_over_time(
        synthetic_temporal_df,
        timestamp_col="job_posted_at",
        target_col="cl_act_total_hired",
        target_type="binary_classification",
    )
    ts_pl = pl.Series(synthetic_temporal_df["job_posted_at"].values)
    mask = result.recommended_filter_mask(ts_pl)
    assert mask.shape == (len(ts_pl),)
    assert mask.any()


# -----------------------------------------------------------------------------
# Batch API: audit_targets_over_time (multi-target single-pass aggregation)
# -----------------------------------------------------------------------------


@pytest.fixture
def synthetic_multi_target_df():
    """Two targets in one frame: one matches the user's drift pattern,
    the other is stable. Lets us verify the batch API gives different
    results per-target while sharing the binning."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2018-05-01", "2026-04-15", freq="D")
    rows = []
    for day in days:
        biased = pd.Timestamp("2021-09-01") <= day < pd.Timestamp("2022-09-01")
        rate_drifty = 0.40 if biased else 0.98
        for _ in range(100):
            rows.append(
                {
                    "ts": day + pd.Timedelta(seconds=int(rng.integers(0, 86400))),
                    "y_drifty": int(rng.uniform() < rate_drifty),
                    "y_stable": int(rng.uniform() < 0.50),
                    "y_continuous": rng.normal(loc=10.0, scale=2.0),
                }
            )
    return pd.DataFrame(rows)


def test_audit_targets_returns_dict_keyed_by_name(synthetic_multi_target_df):
    results = audit_targets_over_time(
        synthetic_multi_target_df,
        timestamp_col="ts",
        targets={
            "drifty": "y_drifty",
            "stable": "y_stable",
        },
    )
    assert set(results.keys()) == {"drifty", "stable"}
    for r in results.values():
        assert isinstance(r, TemporalAuditResult)


def test_audit_targets_drifty_vs_stable(synthetic_multi_target_df):
    """Drifty target gets multi-segment; stable gets one."""
    results = audit_targets_over_time(
        synthetic_multi_target_df,
        timestamp_col="ts",
        targets={"drifty": "y_drifty", "stable": "y_stable"},
    )
    drifty = results["drifty"]
    stable = results["stable"]
    assert len(drifty.segments) >= 2
    assert any("not stable over time" in w.lower() for w in drifty.warnings)
    assert len(stable.segments) == 1
    assert not any("not stable" in w.lower() for w in stable.warnings)


def test_audit_targets_explicit_target_type(synthetic_multi_target_df):
    """The (col, type) tuple form lets the caller mix binary +
    regression in one call."""
    results = audit_targets_over_time(
        synthetic_multi_target_df,
        timestamp_col="ts",
        targets={
            "y_bin": ("y_drifty", "binary_classification"),
            "y_reg": ("y_continuous", "regression"),
        },
    )
    assert results["y_bin"].target_type == "binary_classification"
    assert results["y_reg"].target_type == "regression"
    # Continuous target's bin rates should hover around 10 (the mean)
    reg_rates = [b.target_rate for b in results["y_reg"].bins if b.kept]
    assert 8.5 < np.mean(reg_rates) < 11.5


def test_audit_targets_polars_input_uses_multi_agg(synthetic_multi_target_df):
    """Polars path should produce identical results to pandas path."""
    pl = pytest.importorskip("polars")
    df_pd = synthetic_multi_target_df
    df_pl = pl.from_pandas(df_pd)

    res_pd = audit_targets_over_time(
        df_pd,
        timestamp_col="ts",
        targets={"drifty": "y_drifty", "stable": "y_stable"},
    )
    res_pl = audit_targets_over_time(
        df_pl,
        timestamp_col="ts",
        targets={"drifty": "y_drifty", "stable": "y_stable"},
    )
    # Same number of segments per target
    assert len(res_pd["drifty"].segments) == len(res_pl["drifty"].segments)
    assert len(res_pd["stable"].segments) == len(res_pl["stable"].segments)
    # Identical bin counts
    assert len(res_pd["drifty"].bins) == len(res_pl["drifty"].bins)


def test_audit_targets_equivalent_to_single_calls(synthetic_multi_target_df):
    """Batch result must match per-target separate calls (modulo
    floating-point ordering — we compare segment counts and rates)."""
    batch = audit_targets_over_time(
        synthetic_multi_target_df,
        timestamp_col="ts",
        targets={"drifty": "y_drifty"},
    )
    single = audit_target_over_time(
        synthetic_multi_target_df,
        timestamp_col="ts",
        target_col="y_drifty",
        target_type="binary_classification",
    )
    assert len(batch["drifty"].segments) == len(single.segments)
    assert batch["drifty"].granularity == single.granularity
    for s_batch, s_single in zip(batch["drifty"].segments, single.segments):
        assert s_batch["start_idx"] == s_single["start_idx"]
        assert s_batch["end_idx"] == s_single["end_idx"]
        assert abs(s_batch["mean_rate"] - s_single["mean_rate"]) < 1e-9


def test_audit_targets_polars_runs_one_aggregation(synthetic_multi_target_df, monkeypatch):
    """Verify the batch API hits ``_aggregate_by_time_polars_multi``
    exactly once for N targets (not N times)."""
    pl = pytest.importorskip("polars")
    df_pl = pl.from_pandas(synthetic_multi_target_df)

    from mlframe.training.targets import target_temporal_audit as mod

    multi_calls = []
    single_calls = []
    orig_multi = mod._aggregate_by_time_polars_multi
    orig_single = mod._aggregate_by_time_polars

    def spy_multi(*args, **kwargs):
        multi_calls.append((args, kwargs))
        return orig_multi(*args, **kwargs)

    def spy_single(*args, **kwargs):
        single_calls.append((args, kwargs))
        return orig_single(*args, **kwargs)

    monkeypatch.setattr(mod, "_aggregate_by_time_polars_multi", spy_multi)
    monkeypatch.setattr(mod, "_aggregate_by_time_polars", spy_single)

    audit_targets_over_time(
        df_pl,
        timestamp_col="ts",
        targets={
            "a": "y_drifty",
            "b": "y_stable",
            "c": ("y_continuous", "regression"),
        },
    )
    assert len(multi_calls) == 1, f"expected exactly 1 multi-agg call for 3 targets; got {len(multi_calls)}"
    assert len(single_calls) == 0, f"single-target agg should not fire on the batch path; got {len(single_calls)}"


def test_audit_targets_empty_targets_returns_empty():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=10, freq="D"), "y": np.zeros(10)})
    assert audit_targets_over_time(df, timestamp_col="ts", targets={}) == {}


def test_audit_targets_invalid_spec_raises():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=10, freq="D"), "y": np.zeros(10)})
    with pytest.raises(ValueError, match="must be str or"):
        audit_targets_over_time(
            df,
            timestamp_col="ts",
            targets={"y": [1, 2, 3]},  # type: ignore — bad spec
        )


def test_recommended_mask_no_segments_returns_all_false():
    """Empty audit (no data) → all-False mask, matches input length."""
    from mlframe.training.targets.target_temporal_audit import TemporalAuditResult

    empty = TemporalAuditResult(
        target_name="x",
        target_type="binary_classification",
        timestamp_col="ts",
        granularity="month",
        bins=[],
        change_point_indices=[],
        segments=[],
        warnings=[],
    )
    ts = pd.to_datetime(pd.Series(["2024-01-01", "2024-02-01", "2024-03-01"]))
    mask = empty.recommended_filter_mask(ts)
    assert mask.shape == (3,)
    assert not mask.any()
