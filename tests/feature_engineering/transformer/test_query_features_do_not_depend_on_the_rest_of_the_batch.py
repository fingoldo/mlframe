"""A query row's features must not change with the other rows scored in the same call."""
import numpy as np

from mlframe.feature_engineering.transformer.local_classifier import compute_local_classifier_features
from mlframe.feature_engineering.transformer.mdl_binning_pairwise import compute_mdl_binning_pairwise_features


def _data():
    """A small binary problem and a query batch."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.standard_normal(400) > 0).astype(np.float32)
    return X, y, rng.standard_normal((30, 4)).astype(np.float32)


def test_local_classifier_row_is_the_same_alone_and_in_a_batch():
    """The kernel bandwidth was the median k-th distance over the query batch, so row 0 scored differently alone."""
    X, y, Q = _data()
    batch = compute_local_classifier_features(X, y, Q, seed=1, task="binary").to_numpy()
    alone = compute_local_classifier_features(X, y, Q[:1], seed=1, task="binary").to_numpy()
    assert np.allclose(batch[0], alone[0], atol=1e-5, equal_nan=True)


def test_mdl_binning_row_is_the_same_alone_and_in_a_batch():
    """``unique_combos`` counted the query batch's combinations, so a single row always read 1."""
    X, y, Q = _data()
    batch = compute_mdl_binning_pairwise_features(X, y, Q, seed=1, task="binary").to_numpy()
    alone = compute_mdl_binning_pairwise_features(X, y, Q[:1], seed=1, task="binary").to_numpy()
    assert np.allclose(batch[0], alone[0], atol=1e-5, equal_nan=True)
