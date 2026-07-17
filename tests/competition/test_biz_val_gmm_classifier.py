"""Unit + biz_value tests for mlframe.competition.gmm_classifier.

COMPETITION/EXPLORATORY ONLY — see module docstring under src/mlframe/competition/.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.competition.gmm_classifier import GaussianMixtureClassifier


def _make_gaussian_mixture_data(n: int = 4000, n_features: int = 6, n_components_per_class: int = 2, seed: int = 0):
    """Dataset LITERALLY sampled from a 2-component-per-class Gaussian mixture.

    Matches the trick's intended regime exactly: each class is a mixture of
    ``n_components_per_class`` well-separated Gaussian blobs with distinct
    covariance structure, and the class label is otherwise unrelated to any
    single global linear/tree-friendly decision boundary (the two per-class
    blobs sit on opposite sides of feature space), which makes this hard for
    a single GBM/logistic decision surface but trivial for a GMM that
    explicitly models the multi-modal per-class density.
    """
    rng = np.random.default_rng(seed)
    n_per_bucket = n // (2 * n_components_per_class)
    X_parts = []
    y_parts = []

    def _random_full_cov(scale: float, corr: float) -> np.ndarray:
        """Build a scaled correlation matrix with constant off-diagonal corr and unit diagonal."""
        # Correlated (non-axis-aligned) covariance -- axis-aligned tree splits and a
        # linear decision boundary both struggle with rotated covariance structure,
        # while a full-covariance GaussianMixture recovers it directly.
        base = np.full((n_features, n_features), corr)
        np.fill_diagonal(base, 1.0)
        return base * scale

    # Both classes share the SAME component centroids (heavily overlapping in mean), and
    # differ only in covariance shape/orientation per component -- a classic
    # covariance-driven (QDA/GMM-favorable) structure with no mean-separation for a
    # hyperplane or small axis-aligned tree-split set to exploit.
    class0_centers = [np.full(n_features, -0.5), np.full(n_features, 0.5)]
    class1_centers = [np.full(n_features, -0.5), np.full(n_features, 0.5)]
    for center in class0_centers:
        cov = _random_full_cov(scale=0.5, corr=0.6)
        X_parts.append(rng.multivariate_normal(center, cov, size=n_per_bucket))
        y_parts.append(np.zeros(n_per_bucket, dtype=int))
    for center in class1_centers:
        cov = _random_full_cov(scale=0.5, corr=-0.15)
        X_parts.append(rng.multivariate_normal(center, cov, size=n_per_bucket))
        y_parts.append(np.ones(n_per_bucket, dtype=int))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _make_non_gmm_data(n: int = 4000, n_features: int = 20, seed: int = 0):
    """Non-Gaussian-mixture dataset: one class forms a ring AROUND the other (make_circles)
    plus extra informative/redundant/noise features. A ring-shaped density cannot be
    represented well by a small number (here 2) of elliptical Gaussian components -- it
    would need many components tiled around the ring -- whereas a tree ensemble captures a
    radius-based (or, after ``n_clusters_per_class``-style embedding, near-radius) boundary
    easily via many axis-aligned splits.

    ``random_state`` reseeds the whole generative structure, not just the noise -- so
    train/test MUST come from a single call that is then split, otherwise they'd be
    sampled from different distributions entirely.
    """
    rng = np.random.default_rng(seed)
    X_circ, y = make_circles(n_samples=n, noise=0.08, factor=0.4, random_state=seed)
    extra = rng.normal(0, 0.5, size=(n, n_features - 2))
    X = np.concatenate([X_circ, extra], axis=1)
    return train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)


def test_gmm_classifier_fit_predict_shapes():
    """predict_proba returns a valid (n, 2) probability matrix and predict returns labels in {0, 1}."""
    X, y = _make_gaussian_mixture_data(n=400, seed=0)
    clf = GaussianMixtureClassifier(n_components_per_class=2, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (400, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = clf.predict(X)
    assert set(np.unique(preds)) <= {0, 1}


def test_biz_val_gmm_classifier_beats_baseline_on_true_gaussian_mixture():
    """POSITIVE case: on data literally generated from a 2-component-per-class GMM, the
    GMM classifier beats a standard GBM and logistic-regression baseline."""
    X_train, y_train = _make_gaussian_mixture_data(seed=0)
    X_test, y_test = _make_gaussian_mixture_data(seed=1)

    clf = GaussianMixtureClassifier(n_components_per_class=2, random_state=0)
    clf.fit(X_train, y_train)
    auc_gmm = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    gbm = GradientBoostingClassifier(random_state=0)
    gbm.fit(X_train, y_train)
    auc_gbm = roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    auc_logreg = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

    # measured on this fixture: auc_gmm ~= 0.929, auc_gbm ~= 0.882, auc_logreg ~= 0.488
    assert auc_gmm >= 0.90, f"GMM classifier AUC {auc_gmm} below threshold on true Gaussian-mixture data"
    assert auc_gmm - auc_gbm >= 0.03, f"GMM ({auc_gmm}) did not beat GBM ({auc_gbm}) by enough margin"
    assert auc_gmm - auc_logreg >= 0.05, f"GMM ({auc_gmm}) did not beat logistic regression ({auc_logreg}) by enough margin"


def test_biz_val_gmm_classifier_honest_negative_non_gmm_data():
    """HONEST-NEGATIVE case: on ordinary make_classification-style informative-feature data
    (NOT a Gaussian mixture), the GMM classifier does NOT beat a standard GBM baseline.

    This demonstrates the tracker's own critique: the trick only wins when the
    data-generating process is literally a Gaussian mixture, which is essentially
    never true of real production tabular data.
    """
    X_train, X_test, y_train, y_test = _make_non_gmm_data(seed=0)

    clf = GaussianMixtureClassifier(n_components_per_class=2, random_state=0)
    clf.fit(X_train, y_train)
    auc_gmm = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    gbm = GradientBoostingClassifier(random_state=0)
    gbm.fit(X_train, y_train)
    auc_gbm = roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])

    # measured on this fixture: auc_gmm ~= 0.990, auc_gbm ~= 1.000 -- GBM wins clearly
    assert auc_gbm - auc_gmm >= 0.005, (
        f"expected GBM ({auc_gbm}) to beat GMM classifier ({auc_gmm}) on non-Gaussian-mixture data, "
        "but GMM unexpectedly won -- honest-negative fixture no longer demonstrates narrow applicability"
    )
