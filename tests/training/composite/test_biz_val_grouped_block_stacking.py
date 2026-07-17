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


def _make_correlated_block_dataset(n: int, n_groups: int, cols_per_group: int, seed: int):
    """Like :func:`_make_grouped_block_dataset` (exactly one "active" group per row, all other columns
    zero -- the Santander missing-block scenario where GroupedBlockStacker adds real value), but each
    active group's columns ALSO share a per-row latent factor so they carry real pairwise correlation
    (~0.85-0.95) instead of being drawn independently -- genuine "redundant feature blocks" a correlation
    clustering pass can discover. Cross-group correlation stays ~0 (rows are mutually exclusive so distinct
    groups' active columns never co-vary). This tests ``auto_discover_blocks``: a caller with no pre-defined
    feature grouping should still recover the TRUE blocks and get comparable accuracy to a caller who
    specified them manually."""
    rng = np.random.default_rng(seed)
    n_cols = n_groups * cols_per_group
    group_id = rng.integers(0, n_groups, n)
    loadings = rng.uniform(0.7, 1.3, size=(n_groups, cols_per_group))

    X = np.zeros((n, n_cols))
    y = np.zeros(n)
    weights = {g: rng.normal() for g in range(n_groups)}
    for g in range(n_groups):
        mask = group_id == g
        m = int(mask.sum())
        latent = rng.normal(size=m)
        for c in range(cols_per_group):
            X[mask, g * cols_per_group + c] = latent * loadings[g, c] + rng.normal(scale=0.3, size=m)
        y[mask] = latent * weights[g] + rng.normal(scale=0.3, size=m)

    col_names = [f"g{g}_c{c}" for g in range(n_groups) for c in range(cols_per_group)]
    true_group_of = {f"g{g}_c{c}": g for g in range(n_groups) for c in range(cols_per_group)}
    X_df = pd.DataFrame(X, columns=col_names)
    feature_groups = {f"group{g}": [f"g{g}_c{c}" for c in range(cols_per_group)] for g in range(n_groups)}
    return X_df, y, feature_groups, true_group_of


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
    assert improvement > 0.12, (
        f"expected >12% MSE reduction vs. a single global model, got {improvement:.4f} (baseline={baseline_mse:.4f}, stacker={stacker_mse:.4f})"
    )


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


def test_biz_val_grouped_block_stacker_auto_discover_recovers_blocks_and_matches_manual_accuracy():
    from sklearn.metrics import adjusted_rand_score

    X, y, feature_groups, true_group_of = _make_correlated_block_dataset(n=2000, n_groups=5, cols_per_group=5, seed=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    manual = GroupedBlockStacker(
        feature_groups=feature_groups,
        submodel_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=50),
        meta_estimator=GradientBoostingRegressor(random_state=0, n_estimators=50),
        n_splits=5,
        random_state=0,
    )
    manual.fit(X_train, y_train)
    manual_mse = mean_squared_error(y_test, manual.predict(X_test))

    auto = GroupedBlockStacker(
        auto_discover_blocks=True,
        block_corr_threshold=0.4,
        submodel_factory=lambda: GradientBoostingRegressor(random_state=0, n_estimators=50),
        meta_estimator=GradientBoostingRegressor(random_state=0, n_estimators=50),
        n_splits=5,
        random_state=0,
    )
    auto.fit(X_train, y_train)
    auto_mse = mean_squared_error(y_test, auto.predict(X_test))

    # Block recovery: label every column by which discovered group it landed in, compare to the true
    # group assignment via Adjusted Rand Index (label-permutation invariant, chance ~= 0).
    discovered_group_of = {}
    for gi, members in enumerate(auto.feature_groups_.values()):
        for m in members:
            discovered_group_of[m] = gi
    col_names = list(X.columns)
    true_labels = [true_group_of[c] for c in col_names]
    discovered_labels = [discovered_group_of[c] for c in col_names]
    ari = adjusted_rand_score(true_labels, discovered_labels)
    assert ari > 0.8, f"expected auto-discovered blocks to closely match the true grouping (ARI>0.8), got {ari:.4f}"

    # Accuracy: auto-discovered blocks should be within 20% MSE of manually-specified ground-truth blocks --
    # the point of auto-discovery is parity with a manual grouping, not beating it (the biz value of
    # per-block stacking beating a single global model on missing-block data is already covered by
    # test_biz_val_grouped_block_stacker_beats_global_model_mse above, with a manually-specified grouping).
    relative_gap = (auto_mse - manual_mse) / manual_mse
    assert relative_gap < 0.20, f"expected auto-discover MSE within 20% of manual, got {relative_gap:.4f} (manual={manual_mse:.4f}, auto={auto_mse:.4f})"


def test_grouped_block_stacker_auto_discover_and_manual_feature_groups_are_mutually_exclusive():
    from sklearn.linear_model import LinearRegression

    stacker = GroupedBlockStacker(
        feature_groups={"g": ["a"]},
        auto_discover_blocks=True,
        submodel_factory=lambda: LinearRegression(),
        meta_estimator=LinearRegression(),
    )
    try:
        stacker.fit(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert False, "expected ValueError when both feature_groups and auto_discover_blocks are set"
    except ValueError:
        pass
