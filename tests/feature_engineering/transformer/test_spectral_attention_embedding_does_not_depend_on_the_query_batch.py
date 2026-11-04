"""A query row's spectral embedding is the same alone and inside a batch."""
import numpy as np

from mlframe.feature_engineering.transformer.spectral_attention import compute_spectral_attention


def test_row_scored_alone_equals_the_row_scored_in_a_batch():
    """The Nystrom step took its Gaussian bandwidth from the query batch's own k-th neighbour distances, so batch composition rescaled every embedding."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4)).astype(np.float32)
    Q = rng.standard_normal((25, 4)).astype(np.float32)
    batch = compute_spectral_attention(X, np.zeros(300, dtype=np.float32), Q, seed=1, n_eigvecs=3, k_graph=8).to_numpy()
    alone = compute_spectral_attention(X, np.zeros(300, dtype=np.float32), Q[:1], seed=1, n_eigvecs=3, k_graph=8).to_numpy()
    assert np.allclose(batch[0], alone[0], atol=1e-5, rtol=1e-4)
