"""The surrogate seeder subsamples a large frame before its ten LightGBM fits; a planted two-way interaction is still its first pair, and a frame under the cap is untouched."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._surrogate_interaction_seeder import surrogate_gbm_interaction_seeds


def _needle(n, p=12, seed=0):
    """Ordinal columns with a XOR-like needle between columns 2 and 5 and a binary target."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 6, (n, p)).astype(np.int64)
    y = (((X[:, 2] > 2) ^ (X[:, 5] > 2)) ^ (rng.random(n) < 0.1)).astype(np.int64)
    return X, y


def test_a_capped_large_frame_still_ranks_the_planted_pair_first_and_a_small_one_is_untouched():
    """n = 60000 with a 20000-row cap finds (2, 5); the same call with the cap off and with a cap above n gives identical output."""
    X, y = _needle(60000)
    cols = list(range(X.shape[1]))
    capped, _, info = surrogate_gbm_interaction_seeds(X, y, cols, is_classification=True, row_cap=20000, n_estimators=60)
    assert info["gated"] and capped and tuple(sorted(capped[0])) == (2, 5)
    small_a = surrogate_gbm_interaction_seeds(X[:4000], y[:4000], cols, is_classification=True, row_cap=20000, n_estimators=60)
    small_b = surrogate_gbm_interaction_seeds(X[:4000], y[:4000], cols, is_classification=True, row_cap=0, n_estimators=60)
    assert small_a[0] == small_b[0] and small_a[1] == small_b[1]
