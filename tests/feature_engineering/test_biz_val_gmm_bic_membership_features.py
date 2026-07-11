"""biz_value test for ``feature_engineering.gmm_bic_membership_features.gmm_bic_membership_features``.

The win (8th_instant-gratification.md): a target that depends on which of several latent Gaussian clusters a
row belongs to has a genuinely nonlinear decision surface a linear model can't directly separate from the
raw features, but becomes near-trivially separable once each row's GMM cluster-membership probability is
exposed as a feature. This test also confirms the BIC selection recovers something close to the TRUE
underlying component count, rather than an arbitrary fixed choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.gmm_bic_membership_features import gmm_bic_membership_features


def _make_cluster_driven_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    # 4 well-separated Gaussian clusters in 2D; the label depends on cluster identity (clusters 0,1 -> 0;
    # clusters 2,3 -> 1), a nonlinear decision boundary in raw feature space (clusters interleave positions).
    centers = np.array([[-6, -6], [6, 6], [-6, 6], [6, -6]])
    cluster_labels = np.array([0, 0, 1, 1])
    assignments = rng.integers(0, 4, n)
    X = centers[assignments] + rng.normal(scale=1.5, size=(n, 2))
    y = cluster_labels[assignments]
    return pd.DataFrame(X, columns=["x1", "x2"]), y, assignments


def test_biz_val_gmm_membership_features_linearize_cluster_driven_target():
    df, y, _ = _make_cluster_driven_dataset(n=1500, seed=0)

    auc_raw = roc_auc_score(y, LogisticRegression(max_iter=500).fit(df, y).predict_proba(df)[:, 1])

    membership = gmm_bic_membership_features(df, n_components_range=(2, 3, 4, 5, 6, 8), random_state=0)
    auc_gmm = roc_auc_score(y, LogisticRegression(max_iter=500).fit(membership, y).predict_proba(membership)[:, 1])

    assert auc_raw < 0.75, f"expected a linear model on raw features to struggle with the interleaved-cluster nonlinear boundary, got AUC={auc_raw:.4f}"
    assert auc_gmm > 0.97, f"expected GMM membership-probability features to nearly perfectly linearize the cluster-driven target, got AUC={auc_gmm:.4f}"


def test_gmm_bic_selection_recovers_approximately_true_component_count():
    df, _, assignments = _make_cluster_driven_dataset(n=1500, seed=1)  # 4 true underlying Gaussian clusters
    membership = gmm_bic_membership_features(df, n_components_range=(2, 3, 4, 5, 6, 8), random_state=0)
    selected_k = membership.shape[1]
    assert selected_k in (3, 4, 5), f"expected BIC to select a component count close to the true 4 underlying clusters, got {selected_k}"


def test_gmm_bic_membership_features_rows_sum_to_one():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    membership = gmm_bic_membership_features(df, n_components_range=(2, 3, 4))
    row_sums = membership.sum(axis=1).to_numpy()
    np.testing.assert_allclose(row_sums, np.ones(200), atol=1e-6)
