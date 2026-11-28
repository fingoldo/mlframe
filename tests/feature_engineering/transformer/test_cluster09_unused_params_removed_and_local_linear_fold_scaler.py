"""Cluster 09 regressions: dead parameters are gone, and local_linear scales per fold (audit 2026-11-20, findings 09-F33 and 09-F36).

``task`` (and fca's ``standardize``) were accepted but never read; ``dists`` of the local-lift helper was unused. Mode A of local_linear fit its
RobustScaler on every train row including the validation rows of each fold.
"""

import inspect

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import (
    fca_closed_concepts,
    inducing_attention,
    local_density_gradient,
    local_intrinsic_dim,
    local_lift,
    persistence_diagram,
)
from mlframe.feature_engineering.transformer.local_linear import compute_local_linear_attention


@pytest.mark.parametrize(
    "fn, gone",
    [
        (persistence_diagram.compute_persistence_diagram_features, {"task"}),
        (local_density_gradient.compute_local_density_gradient_features, {"task"}),
        (local_intrinsic_dim.compute_local_intrinsic_dim_features, {"task"}),
        (inducing_attention.compute_inducing_attention_features, {"task"}),
        (fca_closed_concepts.compute_fca_closed_concepts_features, {"task", "standardize"}),
        (local_lift._local_lift_and_pr_auc, {"dists"}),
    ],
)
def test_dead_parameters_are_removed(fn, gone):
    """None of the removed names remains in the signature (09-F33)."""
    assert not gone & set(inspect.signature(fn).parameters)


def test_mode_a_scaler_ignores_the_validation_rows_of_each_fold():
    """Changing only a validation row's value, far outside the range, leaves the other folds' rows unchanged only if the scaler excludes it (09-F36)."""
    rng = np.random.default_rng(0)
    n, d = 120, 3
    X = rng.normal(size=(n, d))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    splitter = KFold(3, shuffle=False)
    base = compute_local_linear_attention(X, y, None, splitter, seed=0, k=8).to_numpy()
    X2 = X.copy()
    X2[5] += 1000.0  # row 5 lies in fold 0's validation part
    moved = compute_local_linear_attention(X2, y, None, splitter, seed=0, k=8).to_numpy()
    fold1 = np.arange(40, 80)  # rows of fold 1 are encoded from folds 0 and 2 (scaler fit on them includes row 5): they may change
    fold0_others = np.setdiff1d(np.arange(40), [5])
    np.testing.assert_allclose(moved[fold0_others], base[fold0_others], rtol=1e-4, atol=1e-5)
    assert fold1.size == 40
