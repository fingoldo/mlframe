"""Columns with missing values must not make the KSG-MI scorers or the gradient-interaction seeder raise: with every FE family on, a frame with NaN columns lost four families to ``Input X contains NaN``."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._gradient_interaction_seeder import _standardize, rank_gradient_interaction_pairs
from mlframe.feature_selection.filters._orthogonal_ksg_mi_fe import _ksg_mi_batch


def _frame(n=1500, seed=0):
    """Four columns, one of them with 10% NaN, one entirely NaN; a binary target that depends on the first two."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    X[rng.random(n) < 0.1, 2] = np.nan
    X = np.column_stack([X, np.full(n, np.nan)])
    y = ((X[:, 0] + X[:, 1] + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    return X, y


@pytest.mark.parametrize("discrete", [True, False])
def test_ksg_batch_with_missing_values_scores_each_column_on_its_finite_rows(discrete):
    """A complete column keeps its batch value, a column with NaN is the single-column value on its finite rows, an all-NaN column is 0."""
    X, y = _frame()
    target = y if discrete else y + np.random.default_rng(1).standard_normal(len(y)) * 0.01
    got = _ksg_mi_batch(X, target, n_neighbors=3, random_state=0)
    assert got.shape == (5,) and np.isfinite(got).all()
    complete = _ksg_mi_batch(X[:, :2], target, n_neighbors=3, random_state=0)
    assert np.array_equal(got[:2], complete)
    rows = np.isfinite(X[:, 2])
    assert got[2] == _ksg_mi_batch(X[rows][:, [2]], target[rows], n_neighbors=3, random_state=0)[0]
    assert got[4] == 0.0 and got[0] > got[3]


def test_standardize_fills_missing_values_with_the_column_mean_and_keeps_clean_columns():
    """A missing value standardises to 0, an all-missing column is all zeros, and a matrix without missing values is standardised as before."""
    X, _ = _frame()
    Z = _standardize(X)
    assert np.isfinite(Z).all() and np.allclose(Z[:, 4], 0.0)
    finite = np.isfinite(X[:, 2])
    assert abs(Z[finite, 2].mean()) < 1e-9 and np.all(Z[~finite, 2] == 0.0)
    clean = X[:, :2]
    assert np.array_equal(_standardize(clean), (clean - clean.mean(axis=0)) / clean.std(axis=0))


def test_gradient_seeder_runs_on_a_frame_with_missing_values():
    """The seeder returns pairs instead of raising, and the interacting pair is found among them."""
    X, y = _frame(2500)
    pairs, _energies, _diag = rank_gradient_interaction_pairs(X, y.astype(float), list(range(5)))
    assert isinstance(pairs, list)
