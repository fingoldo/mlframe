"""The spectral eigenbasis uses the largest algebraic eigenvalues of the normalised adjacency (smallest of the Laplacian), never negative ones."""
import numpy as np
import scipy.sparse as sp

from mlframe.feature_engineering.transformer.spectral_attention import _eigvecs_from_graph


def test_complete_bipartite_graph_gives_no_negative_eigenvalue():
    """K(6, 6) has normalised-adjacency eigenvalues +1, -1 and zeros; 'largest magnitude' returned the -1, which the 1e-6 floor then amplified by 1e6 in the Nystrom step."""
    m = 6
    block = np.ones((m, m), dtype=np.float32)
    W = sp.csr_matrix(np.block([[np.zeros((m, m)), block], [block.T, np.zeros((m, m))]]))
    eigvals, _vecs, _d = _eigvecs_from_graph(W, n_eigvecs=2)
    assert (eigvals >= -1e-6).all(), eigvals
