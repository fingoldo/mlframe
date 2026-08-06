"""FE_TRANSFORMER_B-6: regression quantile edges in y_quintile_baseline_knn had no tie-protection, unlike
the sibling target_quantile.py which bumps non-increasing adjacent edges by 1e-9. A tied/discrete
regression target could make np.quantile produce non-increasing edges, leaving a stratum fully empty and
silently degrading its mean/std features to (0.0, 0.0) instead of a real per-stratum kNN estimate.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import compute_y_quintile_baseline_knn_features

pytestmark = pytest.mark.fast


def test_y_quintile_baseline_knn_no_zero_stratum_on_heavily_tied_regression_target():
    """A heavily-tied (mostly-constant) regression target must not produce any all-zero
    (mean=0, std=0) stratum pair purely from a degenerate/empty quantile bucket."""
    rng = np.random.default_rng(0)
    n, d = 300, 5
    X = rng.standard_normal((n, d)).astype(np.float32)
    # Heavily tied target: only 3 distinct values across 300 rows -- np.quantile on this at 6 edges
    # (5 strata) produces repeated/non-increasing edges without the fix.
    y = rng.choice([1.0, 1.0, 1.0, 1.0, 2.0, 3.0], size=n).astype(np.float32)
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)

    result = compute_y_quintile_baseline_knn_features(X, y, None, splitter, seed=0, task="regression").to_numpy()

    mean_cols = result[:, 0::2]
    std_cols = result[:, 1::2]
    all_zero_stratum = np.all((mean_cols == 0.0) & (std_cols == 0.0), axis=0)
    assert not all_zero_stratum.any(), (
        f"stratum column(s) {np.flatnonzero(all_zero_stratum).tolist()} are all-zero across every row -- "
        "a degenerate/empty quantile bucket from unprotected tied edges."
    )
