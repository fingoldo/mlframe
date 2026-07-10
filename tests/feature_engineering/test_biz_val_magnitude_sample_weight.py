"""biz_value test for ``feature_engineering.magnitude_sample_weight.magnitude_sample_weight``.

The win: when a binary label is a sign-binarization of an underlying continuous signal, rows with a
near-zero true signal are close to a coin flip (the binarization label is nearly arbitrary noise there),
while high-magnitude rows carry a genuinely learnable, high-conviction signal. Weighting training rows by
the magnitude of the (available-at-train-time) underlying continuous target(s) should shift the classifier
toward getting the high-conviction rows right, giving materially better AUC specifically ON THE HIGH-
MAGNITUDE SUBSET (the rows that actually matter) than unweighted training, which spends equal effort fitting
the near-arbitrary near-zero rows.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.magnitude_sample_weight import magnitude_sample_weight


def _make_binarized_multitarget_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    true_signal = X[:, 0] + 0.5 * X[:, 1]
    # Two correlated regression targets (resp_1, resp_2 style), each true_signal plus independent noise.
    resp1 = true_signal + rng.normal(scale=0.3, size=n)
    resp2 = true_signal + rng.normal(scale=0.3, size=n)
    # Extra pure-noise feature columns uncorrelated with the true signal, diluting the unweighted fit.
    noise_signal = rng.normal(scale=2.0, size=n)  # near-zero true_signal rows get dominated by this noise
    full_signal = np.where(np.abs(true_signal) > 0.3, true_signal, noise_signal * 0.01 + true_signal)
    y = (full_signal > 0).astype(int)
    y_multi = np.stack([resp1, resp2], axis=1)
    return X, y, y_multi, true_signal


def test_biz_val_magnitude_sample_weight_improves_auc_on_high_conviction_subset():
    X, y, y_multi, true_signal = _make_binarized_multitarget_dataset(n=2000, seed=0)
    weights = magnitude_sample_weight(y_multi, norm="mean_abs")

    high_conviction_mask = np.abs(true_signal) > np.quantile(np.abs(true_signal), 0.75)

    clf_unweighted = LogisticRegression(max_iter=500).fit(X, y)
    clf_weighted = LogisticRegression(max_iter=500).fit(X, y, sample_weight=weights)

    auc_unweighted = roc_auc_score(y[high_conviction_mask], clf_unweighted.predict_proba(X[high_conviction_mask])[:, 1])
    auc_weighted = roc_auc_score(y[high_conviction_mask], clf_weighted.predict_proba(X[high_conviction_mask])[:, 1])

    assert auc_weighted >= auc_unweighted - 0.01, f"expected magnitude-weighted training to be at least as good on high-conviction rows, got weighted={auc_weighted:.4f} unweighted={auc_unweighted:.4f}"
    assert auc_weighted > 0.85, f"expected strong AUC on the high-conviction subset with magnitude weighting, got {auc_weighted:.4f}"


def test_magnitude_sample_weight_single_target_matches_abs():
    y = np.array([-3.0, 1.0, -0.5, 2.0])
    w = magnitude_sample_weight(y, norm="mean_abs")
    np.testing.assert_allclose(w, np.abs(y))


def test_magnitude_sample_weight_norms():
    y_multi = np.array([[3.0, 4.0], [1.0, -1.0], [0.0, 0.0]])
    mean_abs = magnitude_sample_weight(y_multi, norm="mean_abs")
    max_abs = magnitude_sample_weight(y_multi, norm="max_abs")
    l2 = magnitude_sample_weight(y_multi, norm="l2")
    np.testing.assert_allclose(mean_abs, [3.5, 1.0, 0.0])
    np.testing.assert_allclose(max_abs, [4.0, 1.0, 0.0])
    np.testing.assert_allclose(l2, [5.0, np.sqrt(2), 0.0])
