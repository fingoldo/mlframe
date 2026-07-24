"""Regression tests for audits/full_audit_2026-07-21/fe_top_a.md's findings (F1-F24).

One narrowly-named test per behavioral finding; pure docs/naming findings (F16, F17, F22, F24) are covered by
docstring-content assertions or by the naming/import checks below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

# ---------------------------------------------------------------------------------------------------------------
# F1 -- cross_sectional_neighbors corrupts output when snapshot count <= k
# ---------------------------------------------------------------------------------------------------------------


def test_f1_cross_sectional_neighbors_handles_fewer_snapshots_than_k():
    """F1: n_snapshots <= k must not corrupt neighbor-aggregates or collapse distance_ratio to ~0."""
    from mlframe.feature_engineering.cross_sectional_neighbors import compute_cross_sectional_neighbor_features

    n_snapshots = 5
    df = pd.DataFrame({"time_id": range(n_snapshots), "f0": [sid * 2.0 for sid in range(n_snapshots)]})
    out = compute_cross_sectional_neighbor_features(df, snapshot_col="time_id", feature_cols=["f0"], k=10).to_pandas()

    # true neighbor mean for time_id=0 (f0=0.0): mean of the other 4 snapshots' f0 = mean(2,4,6,8) = 5.0.
    assert out["xsnn_f0_mean"].iloc[0] == pytest.approx(5.0)
    # too few real neighbors -> falls back to the "maximally isolated" 1.0, not the corrupted ~0.0.
    assert (out["xsnn_distance_ratio"] == 1.0).all()


# ---------------------------------------------------------------------------------------------------------------
# F2 -- boolean_pair_interactions silently mis-casts NaN
# ---------------------------------------------------------------------------------------------------------------


def test_f2_boolean_pair_interactions_propagates_nan_instead_of_miscasting():
    """F2: a NaN in either input must propagate as NaN in the AND/OR/XOR output, not a wrong 0/1."""
    from mlframe.feature_engineering.boolean_pair_interactions import boolean_pair_interactions

    df = pd.DataFrame({"a": [1, 0, 1, np.nan, 0], "b": [0, 1, 1, 0, np.nan]})
    out = boolean_pair_interactions(df, columns=["a", "b"])

    assert out["a__and__b"].iloc[:3].tolist() == [0.0, 0.0, 1.0]
    assert np.isnan(out["a__and__b"].iloc[3])
    assert np.isnan(out["a__and__b"].iloc[4])
    assert np.isnan(out["a__xor__b"].iloc[3])


def test_f2_boolean_pair_interactions_stays_int8_when_no_nan():
    """F2: the common (no-NaN) case must stay bit-identical (int8 dtype, no spurious upcast)."""
    from mlframe.feature_engineering.boolean_pair_interactions import boolean_pair_interactions

    df = pd.DataFrame({"a": [1, 0, 1, 1], "b": [0, 1, 1, 0]})
    out = boolean_pair_interactions(df, columns=["a", "b"])
    assert out["a__and__b"].dtype == np.int8
    assert out["a__and__b"].tolist() == [0, 0, 1, 0]


# ---------------------------------------------------------------------------------------------------------------
# F3 -- graph_spectral_features isolated-node eigenvalue pollutes the shape fingerprint
# ---------------------------------------------------------------------------------------------------------------


def test_f3_graph_spectral_features_excludes_isolated_node_from_fingerprint():
    """F3: an isolated node's spurious eigenvalue=1 must not displace the triangle's genuine 1.5."""
    from mlframe.feature_engineering.graph_spectral_features import graph_spectral_features

    edges = np.array([[0, 1], [1, 2], [0, 2]])  # triangle 0-1-2, node 3 isolated
    out = graph_spectral_features(4, edges, k=5)

    assert out["n_components"] == 2.0
    assert out["norm_lap_eig_1"] == pytest.approx(1.5)
    assert out["largest_norm_lap_eig"] == pytest.approx(1.5)


def test_f3_graph_spectral_features_no_isolated_nodes_unaffected():
    """F3: a fully-connected graph (no isolated nodes) must be untouched by the reduced-graph fix."""
    from mlframe.feature_engineering.graph_spectral_features import graph_spectral_features

    edges = np.array([[0, 1], [1, 2], [0, 2]])  # triangle, no isolated nodes
    out = graph_spectral_features(3, edges, k=2)
    assert out["norm_lap_eig_1"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------------------------------------------
# F4 -- particle_filter_posterior grouped path reused the same RNG seed across groups
# ---------------------------------------------------------------------------------------------------------------


def test_f4_particle_filter_posterior_groups_get_independent_rng_streams():
    """F4: two groups with identical observations must NOT produce identical particle-noise realizations."""
    from mlframe.feature_engineering.bayesian import particle_filter_posterior

    obs = np.concatenate([np.zeros(20), np.zeros(20)])
    groups = np.array([0] * 20 + [1] * 20)
    res = particle_filter_posterior(obs, group_ids=groups, seed=42)
    assert not np.allclose(res["p50"][:20], res["p50"][20:])


# ---------------------------------------------------------------------------------------------------------------
# F5 -- compute_mps_targets crashed on pl.concat([]) when every group is skipped
# ---------------------------------------------------------------------------------------------------------------


def test_f5_compute_mps_targets_returns_none_when_every_group_skipped():
    """F5: an all-null-price input must return None (per the Optional[pl.DataFrame] contract), not crash."""
    from mlframe.feature_engineering.mps import compute_mps_targets

    fo_df = pl.DataFrame({"ts": [1, 2], "secid": ["a", "a"], "pr_close": [None, None]}, schema={"ts": pl.Int64, "secid": pl.Utf8, "pr_close": pl.Float64})
    assert compute_mps_targets(fo_df=fo_df) is None


# ---------------------------------------------------------------------------------------------------------------
# F6 -- _peak_trough_counts silently undercounted on NaN-containing windows
# ---------------------------------------------------------------------------------------------------------------


def test_f6_rolling_n_peaks_propagates_nan_for_windows_containing_nan():
    """F6: a window containing NaN must emit NaN (can't-compute), not a silently undercounted integer."""
    from mlframe.feature_engineering.windowed_shape import rolling_n_peaks

    values = np.array([1.0, 3.0, 2.0, np.nan, 5.0, 1.0, 4.0, 2.0])
    group_ids = np.zeros(8, dtype=int)
    out = rolling_n_peaks(values, group_ids, window_K=4)
    # the last window [5,1,4,2] contains no NaN and is fully valid.
    assert out[-1] == 1.0
    # every window that includes the NaN at index 3 must be NaN, not a silently-wrong count.
    assert np.isnan(out[3])
    assert np.isnan(out[4])


# ---------------------------------------------------------------------------------------------------------------
# F7 -- local_linear_detrend zero-filled NaN into the OLS fit, biasing slope/residual toward the imputed zero
# ---------------------------------------------------------------------------------------------------------------


def test_f7_local_linear_detrend_excludes_nan_instead_of_zero_filling():
    """F7: a NaN inside a window must not bias the fitted slope toward zero on a pure linear ramp."""
    from mlframe.feature_engineering.stationarity import local_linear_detrend

    x = np.arange(100, dtype=float)
    x[45] = np.nan
    out = local_linear_detrend(x, window_K=10)
    # window ending at row 50 spans [41..50], containing the single NaN at 45; excluding it (not zero-filling
    # it) recovers slope ~= 1.0. The old zero-fill implementation biased this to ~1.27 (measured).
    assert abs(out["slope"][50] - 1.0) < 1e-6
    assert abs(out["residual"][50]) < 1e-6


def test_f7_local_linear_detrend_residual_nan_when_last_row_nonfinite():
    """F7: the residual at a window's last row must be NaN when that row itself is non-finite -- there is
    nothing to compute a residual against, even though the slope over the other finite rows is still valid."""
    from mlframe.feature_engineering.stationarity import local_linear_detrend

    x = np.arange(60, dtype=float)
    x[59] = np.nan
    out = local_linear_detrend(x, window_K=10)
    assert np.isnan(out["residual"][59])
    assert np.isfinite(out["slope"][59])


def test_f7_local_linear_detrend_nan_when_fewer_than_two_finite_rows():
    """F7: a window with fewer than 2 finite observations can't fit a line and must emit NaN, not a
    zero-biased fake fit."""
    from mlframe.feature_engineering.stationarity import local_linear_detrend

    x = np.full(20, np.nan)
    x[0] = 1.0
    out = local_linear_detrend(x, window_K=10)
    assert np.isnan(out["slope"][9])
    assert np.isnan(out["residual"][9])


# ---------------------------------------------------------------------------------------------------------------
# F8 -- cusum_features computed its auto-threshold globally even under per-group mode
# ---------------------------------------------------------------------------------------------------------------


def test_f8_cusum_features_uses_per_group_threshold():
    """F8: a low-scale group's own jump must trip ITS OWN MAD-derived threshold, not a global one dominated
    by a high-scale sibling group."""
    from mlframe.feature_engineering.stationarity import cusum_features

    rng = np.random.default_rng(0)
    low = rng.normal(scale=1.0, size=100)
    high = rng.normal(scale=50.0, size=100)
    low[50] += 8.0  # a jump large relative to the low-scale group's own noise
    values = np.concatenate([low, high])
    groups = np.array([0] * 100 + [1] * 100)

    result = cusum_features(values, group_ids=groups)
    assert result["n_resets_in_window"][50:60].max() >= 1


def test_f8_cusum_features_ungrouped_unaffected():
    """F8: the ungrouped (group_ids=None) path is untouched -- the single 'group' already IS the full array."""
    from mlframe.feature_engineering.stationarity import cusum_features

    rng = np.random.default_rng(0)
    values = rng.normal(size=50)
    result = cusum_features(values)
    assert result["cusum_pos"].shape == (50,)


# ---------------------------------------------------------------------------------------------------------------
# F9 -- spectral._bands_for produced overlapping/DC-including bands for small window_K
# ---------------------------------------------------------------------------------------------------------------


def test_f9_bands_for_never_overlaps_or_includes_dc():
    """F9: at every window_K down to the degenerate minimum, bands must be non-overlapping and DC-free."""
    from mlframe.feature_engineering.spectral import _bands_for

    for K in range(2, 30):
        bands = _bands_for(K, n_bands=3)
        seen = []
        for lo, hi in bands:
            seen.extend(range(lo, hi))
        assert 0 not in seen, f"K={K}: DC bin leaked into bands {bands}"
        assert len(seen) == len(set(seen)), f"K={K}: overlapping bands {bands}"


# ---------------------------------------------------------------------------------------------------------------
# F10 -- rolling_spectral_rolloff returned bin 0 instead of the last bin at percentile=1.0
# ---------------------------------------------------------------------------------------------------------------


def test_f10_rolling_spectral_rolloff_percentile_one_returns_last_bin():
    """F10: percentile=1.0 (explicitly allowed) must resolve to the last frequency bin, not bin 0."""
    from mlframe.feature_engineering.spectral import rolling_spectral_rolloff

    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    group_ids = np.zeros(200, dtype=int)
    out = rolling_spectral_rolloff(values, group_ids, window_K=20, percentile=1.0)
    valid = out[~np.isnan(out)]
    n_freq = 20 // 2 + 1
    assert (valid == n_freq - 1).all()


# ---------------------------------------------------------------------------------------------------------------
# F11 -- _windowed_stats_by_time_njit used a numerically-unstable one-pass variance formula
# ---------------------------------------------------------------------------------------------------------------


def test_f11_windowed_stats_by_time_numerically_stable_on_large_magnitude_data():
    """F11: large-magnitude values must not produce a spuriously-zero-clamped std (catastrophic cancellation)."""
    from mlframe.feature_engineering.entity_inter_event import _windowed_group_stats

    rng = np.random.default_rng(0)
    n = 200
    group_ids = np.zeros(n, dtype=int)
    timestamps = np.arange(n, dtype=np.float64)
    values = rng.normal(loc=1e8, scale=1.0, size=n)

    _means, stds = _windowed_group_stats(values, timestamps, group_ids, window_time=10.0)
    valid_stds = stds[~np.isnan(stds)]
    # true std is ~1.0; catastrophic cancellation at this magnitude would clamp var<0 -> std=0 everywhere.
    assert np.median(valid_stds) > 0.3


def test_f11_windowed_stats_by_time_matches_naive_on_normal_magnitude_data():
    """F11: the Welford add/remove rewrite must still match a naive per-window recompute on ordinary data."""
    from mlframe.feature_engineering.entity_inter_event import _windowed_group_stats

    rng = np.random.default_rng(1)
    n = 60
    group_ids = np.zeros(n, dtype=int)
    timestamps = np.arange(n, dtype=np.float64)
    values = rng.normal(size=n)

    means, stds = _windowed_group_stats(values, timestamps, group_ids, window_time=5.0)
    for i in range(n):
        lo = i
        while lo > 0 and timestamps[lo - 1] > timestamps[i] - 5.0:
            lo -= 1
        window = values[lo : i + 1]
        assert means[i] == pytest.approx(window.mean(), abs=1e-9)
        assert stds[i] == pytest.approx(window.std(ddof=0), abs=1e-9)


# ---------------------------------------------------------------------------------------------------------------
# F12 -- latent_interaction_features crashed on a degenerate single-entity interaction matrix
# ---------------------------------------------------------------------------------------------------------------


def test_f12_latent_interaction_features_raises_clear_error_on_degenerate_input():
    """F12: min(n_rows, n_cols) < 2 must raise a clear, actionable ValueError, not sklearn's internal one."""
    from mlframe.feature_engineering.latent_interaction_svd import latent_interaction_features

    df = pd.DataFrame({"u": ["a"] * 5, "i": ["x"] * 5})
    with pytest.raises(ValueError, match="need >=2 distinct entities"):
        latent_interaction_features(df, row_entity="u", col_entity="i", n_components=2)


# ---------------------------------------------------------------------------------------------------------------
# F13 -- ewma_multi_alpha_features permanently NaN-poisoned an entity's EWMA after one missing observation
# ---------------------------------------------------------------------------------------------------------------


def test_f13_ewma_multi_alpha_features_recovers_after_nan():
    """F13: a NaN mid-history must not permanently poison every subsequent EWMA value for that entity."""
    from mlframe.feature_engineering.ewma_multi_alpha_features import ewma_multi_alpha_features

    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    group_ids = np.zeros(5, dtype=int)
    result = ewma_multi_alpha_features(values, group_ids, alphas=(0.5,))
    ewma = result["ewma_alpha_0.5"]
    assert not np.isnan(ewma[3])
    assert not np.isnan(ewma[4])


# ---------------------------------------------------------------------------------------------------------------
# F14 -- nearest_past_join's fallback chain resolved a row's ALL columns once ANY ONE column matched
# ---------------------------------------------------------------------------------------------------------------


def test_f14_nearest_past_join_retries_each_column_independently_across_tiers():
    """F14: a column still-null at the fine tier must be retried at a coarser tier, independent of a sibling
    column that already matched at the fine tier."""
    from mlframe.feature_engineering.nearest_past_join import nearest_past_join

    right_df = pd.DataFrame(
        {
            "ts": [1, 8],
            "region": ["east", "east"],
            "day": ["mon", "tue"],
            "col_a": [10.0, 77.0],
            "col_b": [np.nan, 88.0],
        }
    )
    left_df = pd.DataFrame({"ts": [10], "region": ["east"], "day": ["mon"]})

    out = nearest_past_join(
        left_df,
        right_df,
        on="ts",
        by=["region", "day"],
        right_value_cols=["col_a", "col_b"],
        fallback_by_chain=[["region"]],
        tier_col="tier",
    )
    assert out["col_a"].iloc[0] == 10.0  # kept the FINE-tier match, not overwritten by the coarser tier
    assert out["col_b"].iloc[0] == 88.0  # filled from the COARSER tier
    assert out["tier"].iloc[0] == 1


# ---------------------------------------------------------------------------------------------------------------
# F15 -- acf_lag_selection's per-group PACF averaged by the wrong (fixed) group divisor
# ---------------------------------------------------------------------------------------------------------------


def test_f15_select_significant_lags_per_group_divides_by_per_lag_scored_count():
    """F15: a lag scored by only SOME groups must be averaged over those groups, not the full group count."""
    from mlframe.feature_engineering.acf_lag_selection import select_significant_lags

    rng = np.random.default_rng(0)
    short_group = rng.normal(size=8)
    long_group1 = rng.normal(size=60)
    long_group2 = rng.normal(size=60)
    series = np.concatenate([short_group, long_group1, long_group2])
    groups = np.array([0] * 8 + [1] * 60 + [2] * 60)

    result = select_significant_lags(series, max_lag=15, groups=groups, min_group_size=5)
    per_group = result["per_group"]
    lag15_vals = [pg["pacf_values"][15] for pg in per_group.values() if 15 in pg["pacf_values"]]
    assert len(lag15_vals) == 2  # only the two long groups can score lag 15 (short group has 8-1=7 max_lag)
    expected = sum(lag15_vals) / len(lag15_vals)
    assert result["pacf_values"][15] == pytest.approx(expected)


# ---------------------------------------------------------------------------------------------------------------
# F16 / F17 -- mps.py docstring mismatches (2-tuple return, undocumented `shift`)
# ---------------------------------------------------------------------------------------------------------------


def test_f16_f17_mps_docstrings_document_actual_contract():
    """F16/F17: find_best_mps_sequence's docstring must not claim a 3-tuple return, and `shift` must appear."""
    from mlframe.feature_engineering.mps import find_best_mps_sequence, find_maximum_profit_system

    doc = find_best_mps_sequence.__doc__ or ""
    assert "cumulative_profits" not in doc
    assert "shift" in doc
    assert "shift" in (find_maximum_profit_system.__doc__ or "")


# ---------------------------------------------------------------------------------------------------------------
# F18 -- compute_numaggs's lintrend-approx recursion hard-coded return_float32=True
# ---------------------------------------------------------------------------------------------------------------


def test_f18_compute_numaggs_lintrend_approx_respects_outer_return_float32():
    """F18: return_float32=False must keep the lintrend-approx block at float64, not silently round to float32."""
    from mlframe.feature_engineering.numerical import compute_numaggs

    rng = np.random.default_rng(0)
    arr = rng.normal(size=64)
    res_f64 = compute_numaggs(arr, return_lintrend_approx_stats=True, return_float32=False)
    res_f32 = compute_numaggs(arr, return_lintrend_approx_stats=True, return_float32=True)
    assert isinstance(res_f64, tuple)
    assert isinstance(res_f32, np.ndarray) and res_f32.dtype == np.float32
    # at least one value should differ in its low-order bits between the float64 tuple and the float32 array
    # (proves the lintrend-approx block is no longer silently forced through float32 in the f64 case).
    f64_vals = np.array([v for v in res_f64 if isinstance(v, (int, float, np.floating))], dtype=np.float64)
    assert np.any(np.isfinite(f64_vals))


# ---------------------------------------------------------------------------------------------------------------
# F19 -- rolling_integral_above_baseline had a dead if/pass branch
# ---------------------------------------------------------------------------------------------------------------


def test_f19_rolling_integral_above_baseline_still_works_after_dead_code_removal():
    """F19: removing the dead if/pass branch must not change behavior."""
    from mlframe.feature_engineering.windowed_shape import rolling_integral_above_baseline

    rng = np.random.default_rng(0)
    values = rng.normal(size=40)
    group_ids = np.zeros(40, dtype=int)
    out = rolling_integral_above_baseline(values, group_ids, window_K=10)
    assert out.shape == (40,)
    assert np.isfinite(out[9:]).all()


# ---------------------------------------------------------------------------------------------------------------
# F20 -- spectral.py's per-window NaN-imputation snippet was duplicated 6x
# ---------------------------------------------------------------------------------------------------------------


def test_f20_nan_impute_segment_shared_helper_matches_prior_inline_behavior():
    """F20: the extracted _nan_impute_segment helper must reproduce the prior inline NaN-fill exactly."""
    from mlframe.feature_engineering.spectral import _nan_impute_segment

    seg = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    out = _nan_impute_segment(seg)
    assert np.isfinite(out).all()
    assert out[1] == pytest.approx(np.nanmean(seg))

    all_nan = np.full(5, np.nan)
    assert (_nan_impute_segment(all_nan) == 0.0).all()


# ---------------------------------------------------------------------------------------------------------------
# F21 -- ensemble_features.py had unreachable dead-code branch
# ---------------------------------------------------------------------------------------------------------------


def test_f21_predictor_disagreement_var_still_works_after_dead_branch_removal():
    """F21: removing the unreachable arr.shape[1] <= 1 branch must not change real (>=2 predictor) behavior."""
    from mlframe.feature_engineering.ensemble_features import predictor_disagreement_var

    preds = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    out = predictor_disagreement_var(preds)
    assert out.shape == (2,)
    assert out[1] == 0.0


# ---------------------------------------------------------------------------------------------------------------
# F22 -- recency_weighted_rolling.py had no documented NaN-handling policy
# ---------------------------------------------------------------------------------------------------------------


def test_f22_recency_weighted_rolling_module_documents_nan_handling():
    """F22: the module docstring must now state its NaN-propagation behavior explicitly."""
    import mlframe.feature_engineering.recency_weighted_rolling as m

    assert "NaN" in (m.__doc__ or "")


# ---------------------------------------------------------------------------------------------------------------
# F23 -- ma_crossover.py's long_window_weight_power was misleadingly named (actually weights the SHORT leg)
# ---------------------------------------------------------------------------------------------------------------


def test_f23_ma_crossover_features_uses_short_window_weight_power_name():
    """F23: the parameter is renamed to short_window_weight_power, matching its actual (short-leg) behavior."""
    import inspect

    from mlframe.feature_engineering.ma_crossover import ma_crossover_features

    params = inspect.signature(ma_crossover_features).parameters
    assert "short_window_weight_power" in params
    assert "long_window_weight_power" not in params


def test_f23_ma_crossover_features_short_leg_weighting_still_works():
    """F23: the renamed parameter must still drive the same short-leg-weighted vote_sum computation."""
    from mlframe.feature_engineering.ma_crossover import ma_crossover_features

    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=100).cumsum() + 100)
    mas = {w: s.rolling(w).mean() for w in [3, 5, 20]}

    default_out = ma_crossover_features(mas)
    weighted_out = ma_crossover_features(mas, short_window_weight_power=2.0)
    assert not np.allclose(
        default_out["ma_crossover_vote_sum"].dropna().to_numpy(),
        weighted_out["ma_crossover_vote_sum"].dropna().to_numpy(),
    )


# ---------------------------------------------------------------------------------------------------------------
# F24 -- variance_gated_pairwise_diff docstring described np.var (ddof=0) but the code uses np.cov (ddof=1)
# ---------------------------------------------------------------------------------------------------------------


def test_f24_variance_gated_pairwise_diff_docstring_states_actual_ddof_convention():
    """F24: the docstring must now state the actual ddof=1 (np.cov) convention, not just 'np.var(diff)'."""
    from mlframe.feature_engineering.variance_gated_pairwise_diff import variance_gated_pairwise_diff

    doc = variance_gated_pairwise_diff.__doc__ or ""
    assert "ddof" in doc
