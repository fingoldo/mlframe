"""knn_search returns distances in the requested metric from the exact (sklearn) backend, not always Euclidean."""
import numpy as np
from scipy.spatial.distance import cdist

from mlframe.feature_engineering.transformer._knn_helper import knn_search


def _data():
    """A small bank and a few queries."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((200, 6)).astype(np.float32), rng.standard_normal((15, 6)).astype(np.float32)


def test_cosine_metric_returns_cosine_distances():
    """Cosine distance (1 - cos) of the 5 nearest bank rows."""
    xs, xq = _data()
    d, _ = knn_search(xs, xq, 5, metric="cosine", prefer_hnsw_at_n=10**12)
    want = np.sort(cdist(xq, xs, metric="cosine"), axis=1)[:, :5]
    assert np.allclose(d, want, atol=1e-5)


def test_inner_product_metric_returns_one_minus_the_inner_product():
    """hnswlib's 'ip' distance is 1 - <x, q>; the exact backend must agree."""
    xs, xq = _data()
    d, ids = knn_search(xs, xq, 5, metric="ip", prefer_hnsw_at_n=10**12)
    want = np.sort(1.0 - xq @ xs.T, axis=1)[:, :5]
    assert np.allclose(d, want, atol=1e-4)
    assert np.allclose(1.0 - np.einsum("qd,qkd->qk", xq, xs[ids]), d, atol=1e-4)
