"""FE_TRANSFORMER_B-8: k_eff = min(k_neighbors, Xt_s.shape[0]-1) can reach 0 on a single-row train fold;
downstream q_dists[:, k_eff - 1] would then be q_dists[:, -1], a Python negative-index wraparound instead
of a raise, silently producing a numerically-extreme, uninformative log_density with no warning. A
single-row fold must instead return neutral (zero) features with a warning logged.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import LeaveOneOut

from mlframe.feature_engineering.transformer import compute_local_density_gradient_features


def test_local_density_gradient_single_row_train_fold_returns_zeros_with_warning(caplog):
    """LeaveOneOut on 2 rows gives each train fold exactly 1 row (k_eff=0); the function must return
    all-zero features for that fold's row and log a warning, not silently wrap a negative index."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 4)).astype(np.float32)
    y = rng.standard_normal(2).astype(np.float32)
    splitter = LeaveOneOut()

    with caplog.at_level(logging.WARNING, logger="mlframe.feature_engineering.transformer.local_density_gradient"):
        result = compute_local_density_gradient_features(X, y, None, splitter, seed=0, task="regression").to_numpy()

    assert result.shape == (2, 5)
    assert np.all(result == 0.0)
    assert any("too few for any neighbour" in r.message for r in caplog.records)
