"""biz_value test for ``training.composite.GroupedBlockStacker``.

The win: when each row has exactly ONE "active" feature block (all other blocks are zero -- e.g. a sensor
array where only one sensor type reports per row) and each block independently carries real signal about the
target, a single global model trained on the full concatenated (mostly-zero) feature matrix has to learn a
much harder, higher-dimensional problem than it needs to. Fitting one focused submodel per block (using only
that block's own valid rows and its own small feature subset), then stacking their OOF predictions with a
meta-model, should recover a materially lower test MSE -- this mirrors the Santander Value Prediction 2nd
place's 113-groups-of-40 stacking technique.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.training.composite import GroupedBlockStacker


def _make_grouped_block_dataset(n: int, n_groups: int, cols_per_group: int, seed: int):
    rng = np.random.default_rng(seed)
    n_cols = n_groups * cols_per_group
    group_id = rng.integers(0, n_groups, n)

    X = np.zeros((n, n_cols))
    y = np.zeros(n)
    true_w = {g: rng.normal(size=cols_per_group) for g in range(n_groups)}
    for g in range(n_groups):
        mask = group_id == g
        vals = rng.normal(size=(mask.sum(), cols_per_group))
        X[np.ix_(mask, range(g * cols_per_group, (g + 1) * cols_per_group))] = vals
        y[mask] = vals @ true_w[g] + rng.normal(scale=0.3, size=mask.sum())

    col_names = [f"g{g}_c{c}" for g in range(n_groups) for c in range(cols_per_group)]
    X_df = pd.DataFrame(X, columns=col_names)
    feature_groups = {f"group{g}": [f"g{g}_c{c}" for c in range(cols_per_group)] for g in range(n_groups)}
    return X_df, y, feature_groups


def test_biz_val_grouped_block_stacker_beats_global_model_mse():
    X, y, feature_groups = _make_grouped_block_dataset(n=2000, n_groups=6, cols_per_group=6, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    baseline = GradientBoostingRegressor(random_state=0, n_estimators=100)
    baseline.fit(X_train, y_train)
    baseline_mse = mean_squared_error(y_test, baseline.predict(X_test))

    stacker = GroupedBlockStacker(
        feature_groups=feature_groups,
        submodel_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=50),
        meta_estimator=GradientBoostingRegressor(random_state=0, n_estimators=50),
        n_splits=5,
        random_state=0,
    )
    stacker.fit(X_train, y_train)
    stacker_mse = mean_squared_error(y_test, stacker.predict(X_test))

    improvement = 1.0 - stacker_mse / baseline_mse
    assert improvement > 0.12, f"expected >12% MSE reduction vs. a single global model, got {improvement:.4f} (baseline={baseline_mse:.4f}, stacker={stacker_mse:.4f})"


def test_grouped_block_stacker_valid_rates_match_synthetic_group_assignment():
    X, y, feature_groups = _make_grouped_block_dataset(n=1200, n_groups=4, cols_per_group=4, seed=1)
    stacker = GroupedBlockStacker(
        feature_groups=feature_groups,
        submodel_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=20),
        meta_estimator=GradientBoostingRegressor(random_state=0, n_estimators=20),
        n_splits=3,
        random_state=0,
    )
    stacker.fit(X, y)
    # Each row belongs to exactly one of 4 groups, so each group's valid rate should be close to 25%.
    for rate in stacker.group_valid_rates_.values():
        assert 0.15 < rate < 0.35


def test_grouped_block_stacker_requires_nonempty_feature_groups():
    from sklearn.linear_model import LinearRegression

    stacker = GroupedBlockStacker(feature_groups={}, submodel_factory=lambda: LinearRegression(), meta_estimator=LinearRegression())
    try:
        stacker.fit(pd.DataFrame({"a": [1.0, 2.0]}), np.array([1.0, 2.0]))
        assert False, "expected ValueError for empty feature_groups"
    except ValueError:
        pass
