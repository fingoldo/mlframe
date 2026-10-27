"""The cat-pair interaction-information scorer returns the same frame with the O(n) first-seen factorisation as with the sorting ``np.unique`` it replaced."""
import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._mi_greedy_cmi_fe_binning as binning
from mlframe.feature_selection.filters._cat_pair_fe import score_cat_pairs_by_interaction_information


def test_scores_equal_the_np_unique_reference(monkeypatch):
    """Eight categorical columns of mixed cardinality and a target tied to one pair: every pair row (names, joint MI, marginals, II) is identical."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({f"c{i}": rng.integers(0, 3 + 4 * i, n).astype(str) for i in range(8)})
    y = ((X["c1"].astype(int) + X["c2"].astype(int)) % 3 == 0).astype(int).to_numpy()
    new = score_cat_pairs_by_interaction_information(X, y, list(X.columns))
    monkeypatch.setattr(binning, "_factorize_dense_njit", lambda key: (np.unique(key, return_inverse=True)[1].astype(np.int64), 0))
    old = score_cat_pairs_by_interaction_information(X, y, list(X.columns))
    pd.testing.assert_frame_equal(new, old)
    assert new.iloc[0][["cat_i", "cat_j"]].tolist() == ["c1", "c2"]
