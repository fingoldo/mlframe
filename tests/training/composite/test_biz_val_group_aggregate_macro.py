"""biz_value test for ``training.composite.predicted_group_aggregate_feature``.

The win: each cross-sectional group (e.g. ``time_id``) has a shared "macro" component that dominates every
member row's true target, but any SINGLE row's own feature is a very noisy proxy for that shared component
(large idiosyncratic noise per row). Averaging many rows within a group cancels that idiosyncratic noise and
recovers a much cleaner estimate of the shared macro component -- which is exactly what an OOF-safe
group-level auxiliary model does. Feeding its predicted (not realized -- that would leak) group aggregate
back as a per-row feature should let a downstream entity-level model recover the target far better than
using each row's own noisy feature alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

from mlframe.training.composite import predicted_group_aggregate_feature


def _make_macro_panel_dataset(n_groups: int, n_per_group: int, seed: int):
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_groups), n_per_group)
    macro = rng.normal(scale=2.0, size=n_groups)
    n = n_groups * n_per_group
    entity_noise = rng.normal(scale=3.0, size=n)
    x_row = macro[group_ids] + entity_noise
    y_row = macro[group_ids] + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x": x_row}), y_row, group_ids


def test_biz_val_predicted_group_aggregate_feature_beats_entity_only_baseline():
    X, y, group_ids = _make_macro_panel_dataset(n_groups=300, n_per_group=20, seed=0)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    baseline_mse = -cross_val_score(LinearRegression(), X, y, cv=kf, scoring="neg_mean_squared_error").mean()

    macro_feat = predicted_group_aggregate_feature(X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), n_splits=5, random_state=0)
    X_aug = X.copy()
    X_aug["macro"] = macro_feat["predicted_group_aggregate"].to_numpy()
    augmented_mse = -cross_val_score(LinearRegression(), X_aug, y, cv=kf, scoring="neg_mean_squared_error").mean()

    improvement = 1.0 - augmented_mse / baseline_mse
    assert improvement > 0.6, f"expected >60% MSE reduction from adding the predicted macro feature, got {improvement:.4f} (baseline={baseline_mse:.4f}, augmented={augmented_mse:.4f})"


def test_predicted_group_aggregate_feature_output_shape_and_broadcast():
    X, y, group_ids = _make_macro_panel_dataset(n_groups=50, n_per_group=10, seed=1)
    result = predicted_group_aggregate_feature(X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), n_splits=5, random_state=0)
    assert result.shape[0] == X.shape[0]
    # Every row of the SAME group must get the exact same broadcast value.
    df = pd.DataFrame({"group": group_ids, "pred": result["predicted_group_aggregate"].to_numpy()})
    per_group_nunique = df.groupby("group")["pred"].nunique()
    assert (per_group_nunique == 1).all()


def test_predicted_group_aggregate_feature_median_agg():
    X, y, group_ids = _make_macro_panel_dataset(n_groups=60, n_per_group=8, seed=2)
    result_mean = predicted_group_aggregate_feature(X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), agg="mean", n_splits=5, random_state=0)
    result_median = predicted_group_aggregate_feature(X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), agg="median", n_splits=5, random_state=0)
    assert result_mean.shape == result_median.shape
    # mean and median aggregation should differ at least somewhat on noisy data (not bit-identical).
    assert not np.allclose(result_mean["predicted_group_aggregate"].to_numpy(), result_median["predicted_group_aggregate"].to_numpy())


def test_predicted_group_aggregate_feature_default_unchanged_when_aggs_omitted():
    """``aggs`` is opt-in: omitting it must reproduce the exact prior single-``agg`` output, bit-identical."""
    X, y, group_ids = _make_macro_panel_dataset(n_groups=80, n_per_group=10, seed=3)
    result_default = predicted_group_aggregate_feature(X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), n_splits=5, random_state=0)
    result_explicit_agg = predicted_group_aggregate_feature(X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), agg="mean", n_splits=5, random_state=0)
    assert list(result_default.columns) == ["predicted_group_aggregate"]
    assert np.array_equal(result_default["predicted_group_aggregate"].to_numpy(), result_explicit_agg["predicted_group_aggregate"].to_numpy())


def test_biz_val_predicted_group_aggregate_feature_multi_agg_matches_per_call_quality():
    """The opt-in ``aggs=[...]`` multi-statistic path must recover group statistics at least as usefully as
    calling the function once per statistic (single expensive OOF fit shared across all requested stats,
    not a lower-quality shortcut). Prove it on the mean statistic: augmenting with the multi-agg 'mean'
    column must cut downstream MSE by a comparable margin to the existing single-agg biz_value win."""
    X, y, group_ids = _make_macro_panel_dataset(n_groups=300, n_per_group=20, seed=0)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    baseline_mse = -cross_val_score(LinearRegression(), X, y, cv=kf, scoring="neg_mean_squared_error").mean()

    multi = predicted_group_aggregate_feature(
        X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), n_splits=5, random_state=0, aggs=["mean", "median", "std"]
    )
    assert list(multi.columns) == ["predicted_group_aggregate__mean", "predicted_group_aggregate__median", "predicted_group_aggregate__std"]

    X_aug = X.copy()
    X_aug["macro_mean"] = multi["predicted_group_aggregate__mean"].to_numpy()
    augmented_mse = -cross_val_score(LinearRegression(), X_aug, y, cv=kf, scoring="neg_mean_squared_error").mean()

    improvement = 1.0 - augmented_mse / baseline_mse
    assert improvement > 0.6, f"expected >60% MSE reduction from the multi-agg 'mean' column (matching the single-agg win), got {improvement:.4f}"


def test_biz_val_predicted_group_aggregate_feature_multi_agg_speedup_over_per_call():
    """Speed win: one ``aggs=[...]`` call sharing the OOF fit across statistics must be faster than calling
    the function once per statistic (each paying its own K-fold ``LinearRegression`` fit sequence).

    Median-of-N timing per the repo's A/B validation procedure (best-of-N/median, never one-shot) --
    a single perf_counter() sample is noisy enough on a shared/loaded box to flip the verdict. Measured
    median speedup at n_groups=1,500/5,000/20,000 was ~2.08x / ~1.96x / ~1.20x; the threshold here (1.3x)
    sits safely below the small/medium-scale measurement while the win is still large.
    """
    import time

    X, y, group_ids = _make_macro_panel_dataset(n_groups=1_500, n_per_group=20, seed=7)
    aggs = ["mean", "median", "std"]
    n_reps = 5

    def _multi_call() -> None:
        predicted_group_aggregate_feature(X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), n_splits=5, random_state=0, aggs=aggs)

    def _per_stat_calls() -> None:
        for a in aggs:
            predicted_group_aggregate_feature(X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), agg="mean" if a == "std" else a, n_splits=5, random_state=0)

    # Warm up (first-call import/JIT overhead must not pollute the timing).
    _multi_call()
    _per_stat_calls()

    multi_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _multi_call()
        multi_times.append(time.perf_counter() - t0)

    per_stat_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _per_stat_calls()
        per_stat_times.append(time.perf_counter() - t0)

    multi_median_s = float(np.median(multi_times))
    per_stat_median_s = float(np.median(per_stat_times))
    speedup = per_stat_median_s / multi_median_s
    assert speedup > 1.3, (
        f"expected the shared-fit multi-agg call to beat {len(aggs)} separate per-statistic calls by >1.3x "
        f"(median of {n_reps}), got {speedup:.2f}x (multi={multi_median_s*1000:.1f}ms, per-stat={per_stat_median_s*1000:.1f}ms)"
    )
