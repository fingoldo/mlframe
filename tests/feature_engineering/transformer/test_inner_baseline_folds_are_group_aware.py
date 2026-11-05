"""Every transformer that builds a baseline's out-of-fold predictions inside an outer fold gives the inner split the fold's group labels.

A transformer run with ``BoundSplitter(GroupKFold, groups)`` holds out whole groups in the outer split. The inner split that produces the
train-side baseline predictions used to be a shuffled KFold, so the duplicates of a row sat in the training folds of its own baseline and its
out-of-fold prediction read the row's target back. The inner split now follows the same groups.
"""

from __future__ import annotations

import importlib
import logging

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, KFold

from mlframe.feature_engineering.transformer import _baseline_oof
from mlframe.feature_engineering.transformer._utils import BoundSplitter

_FUNCTIONS = {
    "adversarial_flip": "compute_adversarial_flip_features",
    "predictive_info_delta": "compute_predictive_info_delta_features",
    "target_kmeans_codebook": "compute_target_kmeans_codebook_features",
    "variance_baseline": "compute_variance_baseline_features",
    "y_quintile_baseline_knn": "compute_y_quintile_baseline_knn_features",
    "bidir_residual_band": "compute_bidir_residual_band_features",
    "class_balanced_hard_row": "compute_class_balanced_hard_row_features",
    "hard_row_attention": "compute_hard_row_attention_features",
    "multi_temp_cbhr": "compute_multi_temp_cbhr_features",
    "multi_temp_residual_band": "compute_multi_temp_residual_band_features",
    "prediction_band_attention": "compute_prediction_band_attention_features",
    "residual_band_attention": "compute_residual_band_attention_features",
    "signed_residual_band": "compute_signed_residual_band_features",
    "baseline_surprise": "compute_baseline_surprise_features",
    "decision_region_depth": "compute_decision_region_depth_features",
    "disagreement_band": "compute_disagreement_band_features",
    "fisher_weighted_residual": "compute_fisher_weighted_residual_features",
    "ib_baseline_codes": "compute_ib_baseline_codes_features",
    "multi_baseline_hard_row": "compute_multi_baseline_hard_row_features",
    "pairwise_kl_divergence": "compute_pairwise_kl_features",
    "residual_stratified_distance": "compute_residual_stratified_distance_features",
    "sign_residual_baseline": "compute_sign_residual_baseline_features",
}


def _grouped_regression(n_groups: int = 24, rows_per_group: int = 4, n_features: int = 4, seed: int = 0):
    """Rows in groups of ``rows_per_group``, random features and a continuous target."""
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=n)).astype(np.float32)
    return X, y, groups


def test_kfold_splits_without_groups_is_the_shuffled_kfold_the_callers_used():
    """With no group labels the folds are exactly the shuffled KFold the callers built before."""
    X = np.zeros((50, 2))
    expected = list(KFold(n_splits=3, shuffle=True, random_state=17).split(X))
    got = _baseline_oof.kfold_splits(X, 3, 17)
    assert len(got) == len(expected)
    for (tr, va), (etr, eva) in zip(got, expected):
        np.testing.assert_array_equal(tr, etr)
        np.testing.assert_array_equal(va, eva)


def test_kfold_splits_with_groups_never_splits_a_group():
    """With group labels, no group has rows on both sides of any fold."""
    _, _, groups = _grouped_regression()
    X = np.zeros((groups.size, 1))
    for tr, va in _baseline_oof.kfold_splits(X, 3, 0, groups):
        assert set(groups[tr]).isdisjoint(groups[va])


def test_kfold_splits_caps_the_fold_count_at_the_number_of_groups():
    """Two distinct groups give two folds, not three."""
    groups = np.array([0, 0, 1, 1, 0, 1])
    assert len(_baseline_oof.kfold_splits(np.zeros((6, 1)), 3, 0, groups)) == 2


def test_kfold_splits_with_one_group_warns_and_falls_back_to_the_shuffled_kfold(caplog):
    """One distinct group cannot be held out: the shuffled KFold is used and the warning says so."""
    X = np.zeros((12, 1))
    with caplog.at_level(logging.WARNING, logger=_baseline_oof.logger.name):
        got = _baseline_oof.kfold_splits(X, 3, 5, np.zeros(12, dtype=int))
    expected = list(KFold(n_splits=3, shuffle=True, random_state=5).split(X))
    assert "cannot be held out" in caplog.text
    for (tr, va), (etr, eva) in zip(got, expected):
        np.testing.assert_array_equal(tr, etr)
        np.testing.assert_array_equal(va, eva)


@pytest.mark.parametrize("module_name", sorted(_FUNCTIONS))
def test_transformer_inner_baseline_folds_follow_the_outer_groups(module_name, monkeypatch):
    """Run a transformer with a group-bound outer splitter: every inner baseline split gets the labels and holds whole groups out."""
    calls = []
    real = _baseline_oof.kfold_splits

    def spy(X, n_splits, random_state, groups=None):
        """Record the group labels and the folds of every inner split, then return the real folds."""
        out = real(X, n_splits, random_state, groups)
        calls.append((groups, out))
        return out

    monkeypatch.setattr(_baseline_oof, "kfold_splits", spy)
    X, y, groups = _grouped_regression()
    fn = getattr(importlib.import_module(f"mlframe.feature_engineering.transformer.{module_name}"), _FUNCTIONS[module_name])
    fn(X_train=X, y_train=y, X_query=None, task="regression", splitter=BoundSplitter(GroupKFold(n_splits=2), groups=groups), seed=0)

    assert calls, f"{module_name} never built an inner baseline split"
    for call_groups, splits in calls:
        assert call_groups is not None, f"{module_name} built an inner split without the group labels"
        labels = np.asarray(call_groups)
        for tr, va in splits:
            assert set(labels[tr]).isdisjoint(labels[va])
