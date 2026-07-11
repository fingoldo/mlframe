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


def test_gmm_bic_membership_features_new_df_none_is_bit_identical_to_prior_behavior():
    # Regression test: the opt-in `new_df` parameter must not change default behavior at all.
    df, _, _ = _make_cluster_driven_dataset(n=300, seed=3)
    baseline = gmm_bic_membership_features(df, n_components_range=(2, 3, 4, 5), random_state=0)
    extended = gmm_bic_membership_features(df, n_components_range=(2, 3, 4, 5), random_state=0, new_df=None)
    pd.testing.assert_frame_equal(baseline, extended)
    assert baseline.attrs == {}


def test_biz_val_gmm_membership_features_detects_train_test_distribution_shift():
    # The win: fit the GMM on in-distribution data, then score a batch drawn from a materially different
    # region of feature space. Without a diagnostic, the returned membership probabilities look like
    # ordinary features with no signal that the GMM's density model no longer describes this data --
    # a silent-garbage failure mode. `gmm_shift_diagnostics` must flag this batch and NOT flag an
    # in-distribution batch drawn from the same generating process as training.
    df, _, _ = _make_cluster_driven_dataset(n=1500, seed=4)

    rng = np.random.default_rng(5)
    in_distribution_new, _, _ = _make_cluster_driven_dataset(n=300, seed=6)
    shifted_new = pd.DataFrame(rng.normal(loc=40, scale=1.5, size=(300, 2)), columns=["x1", "x2"])  # far outside all 4 training clusters

    in_dist_result = gmm_bic_membership_features(df, n_components_range=(2, 3, 4, 5, 6, 8), random_state=0, new_df=in_distribution_new)
    shifted_result = gmm_bic_membership_features(df, n_components_range=(2, 3, 4, 5, 6, 8), random_state=0, new_df=shifted_new)

    in_dist_diag = in_dist_result.attrs["gmm_shift_diagnostics"]
    shifted_diag = shifted_result.attrs["gmm_shift_diagnostics"]

    assert in_dist_diag["distribution_shift_detected"] is False, f"expected no shift flag on in-distribution new data, got diagnostics={in_dist_diag}"
    assert shifted_diag["distribution_shift_detected"] is True, f"expected a shift flag on far-out-of-distribution new data, got diagnostics={shifted_diag}"
    assert shifted_diag["shift_zscore"] > 20, f"expected a large shift z-score for badly out-of-distribution data, got {shifted_diag['shift_zscore']:.2f}"
