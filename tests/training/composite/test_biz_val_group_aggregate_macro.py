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
