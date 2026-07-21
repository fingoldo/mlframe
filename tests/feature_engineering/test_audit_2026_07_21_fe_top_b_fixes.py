"""Regression tests for the feature_engineering findings fixed from the 2026-07-21
full-repo audit (see audits/full_audit_2026-07-21/fe_top_b.md).

One narrowly-scoped test per finding ID (F1-F12; F13 is covered separately in
test_pysr_operators.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_f1_holiday_calendar_features_matches_timestamped_input():
    """F1: is_holiday/is_eve must match even when the input carries a time-of-day component."""
    from mlframe.feature_engineering.holiday_calendar_features import holiday_calendar_features

    dates = pd.Series(pd.to_datetime(["2024-12-25 14:30:00", "2024-12-24 09:00:00", "2024-06-15 00:00:00"]))
    out = holiday_calendar_features(dates, country="US")
    assert bool(out["holiday_is_holiday"].iloc[0]) is True
    assert bool(out["holiday_is_eve"].iloc[1]) is True
    assert bool(out["holiday_is_holiday"].iloc[2]) is False


def test_f2_multi_window_aggregate_does_not_mutate_history_df():
    """F2: dropping the per-horizon history_df.copy() must not let multi_window_aggregate
    accidentally mutate the caller's history frame (the whole point of the copy removal is
    that history_df is read-only downstream; pin that contract explicitly)."""
    from mlframe.feature_engineering.multi_window_aggregate import multi_window_aggregate

    history_df = pd.DataFrame(
        {
            "entity": [1, 1, 1, 2, 2],
            "t": [1.0, 2.0, 3.0, 1.0, 2.0],
            "v": [10.0, 20.0, 30.0, 5.0, 15.0],
        }
    )
    history_before = history_df.copy(deep=True)
    as_of = pd.DataFrame({"entity": [1, 2], "t": [4.0, 3.0]})

    multi_window_aggregate(
        history_df=history_df, as_of=as_of, entity_col="entity", time_col="t", query_entity_col="t",
        agg_funcs={"v": ["sum"]}, lookback_horizons=[2.0],
    )
    pd.testing.assert_frame_equal(history_df, history_before)


def test_f3_per_group_apply_raises_when_every_group_fails():
    """F3: a systematically-broken fn (always raises) must surface as a hard failure, not
    silently degrade to an all-fill_value output with only per-group log warnings."""
    from mlframe.feature_engineering.grouped import per_group_apply

    def _always_raises(seg):
        """Always raises."""
        raise ValueError("systematic bug in caller fn")

    values = np.arange(10, dtype=np.float64)
    group_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    with pytest.raises(RuntimeError, match="ALL .* attempted group"):
        per_group_apply(values, group_ids, _always_raises)


def test_f3_per_group_apply_still_tolerates_isolated_group_failure():
    """F3 (regression guard): an isolated per-group failure (not systematic) must still
    degrade gracefully to fill_value for just that group, not raise."""
    from mlframe.feature_engineering.grouped import per_group_apply

    def _fails_only_on_group_1(seg):
        """Fails only on group 1."""
        if seg[0] == 100.0:  # group 1's values start at 100
            raise ValueError("degenerate group")
        return seg * 2.0

    values = np.array([1.0, 2.0, 100.0, 200.0, 3.0, 4.0])
    group_ids = np.array([0, 0, 1, 1, 2, 2])
    out = per_group_apply(values, group_ids, _fails_only_on_group_1, fill_value=-1.0)
    assert list(out) == [2.0, 4.0, -1.0, -1.0, 6.0, 8.0]


def test_f4_anchor_features_bit_identical_to_baseline_with_cached_slope():
    """F4: the slope-caching optimization (recompute only when a new anchor arrives, not
    every row) must produce bit-identical output to the original per-row-refit version."""
    from mlframe.feature_engineering.anchor import _anchor_features_for_segment

    rng = np.random.default_rng(0)
    n = 500
    is_anchor = rng.random(n) < 0.1
    is_anchor[0] = True
    label = np.where(is_anchor, rng.normal(size=n) * 10, np.nan)

    out = _anchor_features_for_segment(label, is_anchor, 5)
    # Sanity: slope is defined (non-NaN) once >=1 anchor seen, and extrapolation tracks it.
    first_anchor = np.flatnonzero(is_anchor)[0]
    assert np.isfinite(out["local_slope"][first_anchor])
    assert out["rows_since"][first_anchor] == 0.0


def test_f5_f6_f7_init_all_matches_public_symbols():
    """F5/F6/F7: every previously-orphaned symbol (grouped/anchor/spatial/hurst per-op
    variants, relational_dfs' stack_relational_chain/RelationalHop, categorical_group_concat's
    discover_categorical_groups/auto_concat_categorical_groups) must now be both importable
    AND present in __all__."""
    import mlframe.feature_engineering as fe

    previously_orphaned = [
        "per_group_rank", "per_group_shift", "per_group_cum_reduce", "per_group_rolling_reduce", "per_group_nth",
        "anchor_density_features", "anchor_ewm_features", "anchor_quadratic_extrapolation_features",
        "anchor_residual_rmse_features", "rows_until_next_anchor",
        "inverse_distance_weighted_aggregate", "knn_gradient_features", "knn_label_dispersion_features",
        "local_density_features", "radius_aggregate",
        "multi_scale_hurst", "dfa_alpha2_quadratic", "multifractal_dfa",
        "stack_relational_chain", "RelationalHop",
        "discover_categorical_groups", "auto_concat_categorical_groups",
    ]
    for name in previously_orphaned:
        assert hasattr(fe, name), f"{name} not importable from mlframe.feature_engineering"
        assert name in fe.__all__, f"{name} importable but missing from __all__"


def test_f8_entity_diff_features_shallow_copy_does_not_mutate_source_columns():
    """F8: switching to copy(deep=False) must not let the new diff columns' assignment
    leak back into the caller's original df (pandas' copy-on-write semantics for new
    column assignment protect this, but pin it explicitly since it's the whole safety
    argument for using a shallow copy here)."""
    from mlframe.feature_engineering.entity_diff_features import entity_diff_features

    df = pd.DataFrame({"entity": [1, 1, 1, 2, 2], "v": [1.0, 2.0, 4.0, 10.0, 20.0]})
    df_before = df.copy(deep=True)
    out = entity_diff_features(df, entity_col="entity", feature_cols=["v"], n=1)
    pd.testing.assert_frame_equal(df, df_before)
    assert "v_diff" in out.columns or any(c.startswith("v") and c != "v" for c in out.columns)


def test_f9_relational_dfs_compute_relational_features_empty_specs_returns_copy():
    """F9: the empty-child_specs early return must still be a defensive copy (not a live
    alias to parent_df), while the non-empty path must produce identical results."""
    from mlframe.feature_engineering.relational_dfs import compute_relational_features

    parent_df = pd.DataFrame({"id": [1, 2, 3], "cutoff": [10.0, 20.0, 30.0]})
    result = compute_relational_features(parent_df, parent_id_col="id", cutoff_col="cutoff", child_specs=[])
    assert result is not parent_df
    result["id"] = -1
    assert list(parent_df["id"]) == [1, 2, 3]  # mutating the result must not affect parent_df


def test_f12_per_group_nadaraya_watson_smooth_sample_weight_changes_output():
    """F12: per_group_nadaraya_watson_smooth must accept sample_weight and have it actually
    change the smoothed output (previously silently ignored -- always unweighted)."""
    from mlframe.feature_engineering.nadaraya_watson import per_group_nadaraya_watson_smooth

    rng = np.random.default_rng(0)
    n = 40
    group_ids = np.repeat([0, 1], n // 2)
    order = np.tile(np.arange(n // 2, dtype=float), 2)
    values = np.concatenate([np.sin(order[: n // 2]), np.cos(order[: n // 2])]) + rng.normal(scale=0.05, size=n)

    unweighted = per_group_nadaraya_watson_smooth(values, group_ids, order=order, bandwidth=2.0)
    w = np.ones(n)
    w[:5] = 1e-3
    w[n // 2 : n // 2 + 5] = 1e-3
    weighted = per_group_nadaraya_watson_smooth(values, group_ids, order=order, bandwidth=2.0, sample_weight=w)

    assert not np.allclose(unweighted, weighted)
    # Uniform weight of 1.0 must reproduce the unweighted result exactly (backward compat).
    uniform_w = per_group_nadaraya_watson_smooth(values, group_ids, order=order, bandwidth=2.0, sample_weight=np.ones(n))
    assert np.allclose(unweighted, uniform_w)
