"""Performer features of a query row do not change with the other rows scored beside it, and survive a far-away query."""
import numpy as np

from mlframe.feature_engineering.transformer.performer_attention import compute_performer_attention_features


def _data():
    """A training set and a few query rows."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 6)).astype(np.float32)
    y = (X[:, 0] + 0.2 * rng.standard_normal(300)).astype(np.float32)
    return X, y, rng.standard_normal((10, 6)).astype(np.float32)


def test_row_alone_equals_the_row_inside_a_batch():
    """The shift was the global max over the query batch, so kernel_conc and the log normalizer of a row depended on its batch companions."""
    X, y, Q = _data()
    batch = compute_performer_attention_features(X, y, Q, seed=1, standardize=False).to_numpy()
    alone = compute_performer_attention_features(X, y, Q[:1], seed=1, standardize=False).to_numpy()
    assert np.allclose(batch[0], alone[0], rtol=1e-4, atol=1e-5)


def test_far_query_does_not_collapse_to_a_zero_estimate():
    """A query whose features sit far below the key-side maximum underflowed float32 and read y_est = 0 through the 1e-9 floor."""
    X, y, _q = _data()
    far = np.full((1, 6), 8.0, dtype=np.float32)
    out = compute_performer_attention_features(X, y, far, seed=1, standardize=False).to_numpy()
    assert np.isfinite(out).all() and float(out[0, 0]) != 0.0, out
