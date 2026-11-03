"""Fold vote weights of the per-fold stability selection: NaN fidelity is neutral, all-zero falls back to equal votes."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_compose import fold_vote_weights


def test_nan_fidelity_fold_gets_the_mean_of_the_finite_folds():
    """Folds [0.8, NaN, 0.6]: the NaN fold is weighted 0.7, not silenced."""
    w = fold_vote_weights([0.8, float("nan"), 0.6], True)
    np.testing.assert_allclose(w, [0.8, 0.7, 0.6])


def test_all_nan_or_non_positive_fidelities_fall_back_to_equal_votes():
    """Every fold NaN (single-class holdouts) used to give total weight 0 and an empty ensemble."""
    assert fold_vote_weights([float("nan")] * 3, True) == [1.0, 1.0, 1.0]
    assert fold_vote_weights([-0.2, 0.0, -0.5], True) == [1.0, 1.0, 1.0]


def test_unweighted_mode_is_one_fold_one_vote():
    """weight_by_fidelity=False ignores fidelity (negative control)."""
    assert fold_vote_weights([0.1, float("nan")], False) == [1.0, 1.0]
