"""The Nystrom extension applies d_i^{-1/2} to each neighbour exactly once (audit 2026-11-20, finding 09-F2)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from mlframe.feature_engineering.transformer.spectral_attention import (
    _build_knn_graph,
    _eigvecs_from_graph,
    _nystrom_extend,
)


def test_nystrom_matches_the_documented_single_degree_scaling():
    """Reference: phi_j(q) = (1/lambda_j) * d_q^{-1/2} * sum_i w(q,i) * d_i^{-1/2} * u_j[i], with u_j the eigenvectors of D^{-1/2} W D^{-1/2}.

    The extension used to scale the eigenvectors by d_i^{-1/2} a second time, which mis-weights every neighbour whose degree is not 1.
    """
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((40, 2))
    X_query = rng.standard_normal((5, 2))
    k_graph, sigma, n_eig = 6, 1.0, 3

    W, _ids, _dists = _build_knn_graph(X_train, k_graph, sigma=sigma)
    eigvals, eigvecs, d_inv_sqrt = _eigvecs_from_graph(W, n_eigvecs=n_eig)
    got = _nystrom_extend(X_train, X_query, eigvals=eigvals, eigvecs_normalised=eigvecs, d_inv_sqrt=d_inv_sqrt, k_graph=k_graph, sigma=sigma)

    dists, ids = NearestNeighbors(n_neighbors=k_graph).fit(X_train).kneighbors(X_query)
    aff = np.exp(-(dists**2) / (2 * sigma**2))
    d_q = aff.sum(axis=1) + 1e-9
    expected = np.zeros((X_query.shape[0], n_eig))
    for q in range(X_query.shape[0]):
        raw = (aff[q][:, None] * d_inv_sqrt[ids[q]][:, None] * eigvecs[ids[q]]).sum(axis=0)
        expected[q] = raw / np.sqrt(d_q[q]) / np.maximum(eigvals, 1e-6)

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
