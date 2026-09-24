"""A grouped transform is scored by the tiny-model CV with its groups on every forward, inverse and per-fold refit.

The CV computed T and reconstructed y with bare ``transform.forward`` / ``inverse`` calls that never received the rows'
groups, so a grouped spec raised before any fold ran and dropped out of the rerank, even when too few groups forced the split
itself back to plain KFold.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._screening_tiny_perbin import _tiny_cv_rmse_y_scale
from mlframe.training.composite.transforms import get_transform


@pytest.mark.parametrize("n_groups", [2, 8])
def test_a_grouped_transform_gets_a_finite_tiny_cv_score(n_groups: int):
    """With 8 groups the split is grouped; with 2 (< 3 folds) it falls back to KFold, and the transform still gets its groups."""
    rng = np.random.default_rng(0)
    n = 400
    g = np.arange(n) % n_groups
    base = rng.uniform(1.0, 10.0, n)
    x = rng.normal(size=(n, 2))
    y = (1.0 + 0.3 * g) * base + x[:, 0] + rng.normal(0.0, 0.2, n)
    t = get_transform("linear_residual_grouped")
    params = t.fit(y, base, groups=g)
    rmse = _tiny_cv_rmse_y_scale(y, base, t, params, x, family="lightgbm", n_estimators=10, num_leaves=7, learning_rate=0.1,
                                 cv_folds=3, random_state=0, groups=g)
    assert np.isfinite(rmse) and rmse < float(np.std(y)), rmse
