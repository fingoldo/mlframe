"""The top-k proximity neighbours equal a full sort's top-k, without negating the similarity block (audit 2026-11-20, finding 09-F10)."""
import numpy as np

from mlframe.feature_engineering.transformer.rf_proximity import _topk_proximity


def test_topk_equals_the_full_sort_top_k():
    """Random similarities: the returned ids are the k largest, in descending order, with their similarities."""
    rng = np.random.default_rng(0)
    n_query, n_bank, k = 15, 40, 5
    S_query = rng.random((n_query, 8)).astype(np.float32)
    S_bank = rng.random((n_bank, 8)).astype(np.float32)
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize
    Sq = normalize(csr_matrix(S_query))
    Sb = normalize(csr_matrix(S_bank))
    ids, sims = _topk_proximity(Sq, Sb, k)
    full = (Sq @ Sb.T).toarray()
    expected_ids = np.argsort(-full, axis=1, kind="stable")[:, :k]
    np.testing.assert_array_equal(np.sort(ids, axis=1), np.sort(expected_ids, axis=1))
    np.testing.assert_allclose(sims, np.take_along_axis(full, ids, axis=1), rtol=1e-5)
