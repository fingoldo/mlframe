"""Cluster 09 regression: the three-member baseline ensemble has one definition shared by its five users (audit 2026-11-20, finding 09-F31)."""

import numpy as np

import mlframe.feature_engineering.transformer.disagreement_band as db
import mlframe.feature_engineering.transformer.ib_baseline_codes as ib
import mlframe.feature_engineering.transformer.multi_baseline_hard_row as mb
from mlframe.feature_engineering.transformer import _baseline_oof as bo


def _data(n=120, seed=0):
    """A small binary problem with one informative column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] + 0.4 * rng.normal(size=n) > 0).astype(np.float32)
    return X, y


def test_users_return_the_shared_out_of_fold_ensemble_in_their_own_shape():
    """disagreement_band gets the (n, 3) array, multi_baseline_hard_row the same columns as a list, ib_baseline_codes the same train side."""
    X, y = _data()
    shared = bo.three_member_oof(X, y, "binary", 5, None, "test")
    np.testing.assert_array_equal(db._fit_3baselines_oof(X, y, "binary", 5), shared)
    cols = mb._fit_3baselines_predict(X, y, "binary", 5)
    np.testing.assert_array_equal(np.column_stack(cols), shared)
    train_side, query_side = ib._fit_3baselines_two(X, y, X[:10], "binary", 5)
    np.testing.assert_array_equal(train_side, shared)
    assert query_side.shape == (10, 3)


def test_a_single_class_target_falls_back_to_the_class_prior_instead_of_raising():
    """All-one labels give the prior (1.0) for every member rather than a LightGBM error."""
    X, _ = _data()
    out = bo.three_member_predictions(X, np.ones(len(X), dtype=np.float32), X[:5], "binary", 1, "test")
    assert (out == 1.0).all() and out.shape == (5, 3)
