"""The weak-segment diagnostic fitted an exact sklearn tree only to RANK columns.

sklearn's exact splitter sorts every feature at every node to do that, which made the fit the single
largest cost in the reporting package. Ranking does not need an exact tree: what the chart asks for is
"which columns, split somewhere, separate the high-error rows", and that is each column's depth-1 variance
reduction, readable off a quantile-binned (count, error-sum) histogram in one pass per column.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts._split_gain_ranking import (
    RUNNER_UP_MIN_SHARE,
    rank_features_by_split_gain,
    split_gain_per_feature,
    top_by_gain,
)


def _planted(n: int = 20_000, p: int = 40, n_signal: int = 3, seed: int = 0):
    """A noise matrix with `n_signal` columns that really do separate the error, strongest first."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(n, p))
    err = rng.normal(size=n) * 0.1
    signal = sorted({int(rng.integers(0, p)) for _ in range(n_signal)})
    for k, j in enumerate(signal):
        err += (mat[:, j] > 0.5) * (2.0 - 0.5 * k)
    return mat, err, signal


def test_the_planted_features_come_out_on_top():
    """The three columns that really separate the error are the three that come back."""
    mat, err, signal = _planted()
    assert set(rank_features_by_split_gain(mat, err, len(signal))) == set(signal)


@pytest.mark.parametrize("seed", range(6))
def test_it_recovers_at_least_as_much_as_the_tree_it_replaced(seed):
    """The bar for replacing the tree is not "close enough" but "finds the same signal"."""
    sklearn_tree = pytest.importorskip("sklearn.tree")
    mat, err, signal = _planted(seed=seed)
    ours = set(rank_features_by_split_gain(mat, err, len(signal)))
    tree = sklearn_tree.DecisionTreeRegressor(max_depth=3, random_state=0).fit(mat, err)
    theirs = {int(j) for j in np.argsort(tree.feature_importances_)[::-1][: len(signal)]}
    assert len(ours & set(signal)) >= len(theirs & set(signal)), f"we found {sorted(ours)}, the tree found {sorted(theirs)}, planted {signal}"


def test_a_noise_column_is_not_promoted_beside_a_real_one():
    """EVERY column has a positive best-split gain -- the luckiest cut in pure noise separates something.

    Reporting on that would give the heatmap a second axis the data does not have; the tree assigned those
    columns an importance of exactly zero and drew a 1-D grid. This is the guard that keeps that behaviour.
    """
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(20_000, 40))
    err = rng.normal(size=20_000) * 0.1 + (mat[:, 7] > 0.5) * 3.0
    gains = split_gain_per_feature(mat, err)
    assert (gains[np.arange(40) != 7] > 0).any(), "the fixture is not exercising the guard: no noise column scores at all"
    assert rank_features_by_split_gain(mat, err, 2) == [7], "a noise column was promoted alongside the one real separator"


def test_two_real_features_are_both_reported():
    """Guard: the runner-up floor must not suppress a genuine second dimension."""
    rng = np.random.default_rng(2)
    mat = rng.normal(size=(20_000, 40))
    err = rng.normal(size=20_000) * 0.1 + (mat[:, 3] > 0.5) * 3.0 + (mat[:, 11] > 0.5) * 2.5
    assert set(rank_features_by_split_gain(mat, err, 2)) == {3, 11}


def test_the_floor_is_relative_to_the_winner():
    """Directly on the selector, so the threshold's meaning is pinned independently of any fixture."""
    gains = np.array([1.0, RUNNER_UP_MIN_SHARE * 1.01, RUNNER_UP_MIN_SHARE * 0.99])
    assert top_by_gain(gains, 3) == [0, 1], "the floor is not being applied as a share of the best gain"


def test_a_column_of_one_value_scores_nothing():
    """A constant column cannot be split, and must not score above a column that can."""
    rng = np.random.default_rng(3)
    mat = np.column_stack([np.full(5_000, 4.2), rng.normal(size=5_000)])
    err = rng.normal(size=5_000) + (mat[:, 1] > 0.0) * 2.0
    gains = split_gain_per_feature(mat, err)
    assert gains[0] == 0.0, f"a constant column scored {gains[0]}"
    assert gains[1] > 0.0


def test_missingness_can_itself_be_the_signal():
    """A median split silently dropped this case; the non-finite rows get their own bin."""
    rng = np.random.default_rng(4)
    col = rng.normal(size=10_000)
    missing = rng.random(10_000) < 0.3
    col[missing] = np.nan
    mat = np.column_stack([col, rng.normal(size=10_000)])
    err = rng.normal(size=10_000) * 0.1 + missing * 3.0
    assert rank_features_by_split_gain(mat, err, 1) == [0], "the column whose MISSINGNESS carries the error was not found"
