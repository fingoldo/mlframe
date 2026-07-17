"""Regression test: ``sign_residual_baseline``'s 5th output column (``{prefix}_baseline_score``) must be a
bias-corrected point estimate distinct from ``{prefix}_mu``, not a duplicate of it.

Pre-fix, ``_process`` returned ``np.column_stack([mu_query, p_pos, bias_signal, abs_bias, mu_query])`` — the 5th
column was an exact copy of the 1st, wasting a feature slot instead of encoding the bias-adjusted prediction the
module's docstring describes.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer.sign_residual_baseline import compute_sign_residual_baseline_features


def test_baseline_score_differs_from_mu_when_bias_detected():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 4)).astype(np.float32)
    # Asymmetric residual: true y is systematically above a linear baseline for a subset of rows,
    # driven by feature 0 - this is the directional bias the sign-classifier should detect.
    base = X[:, 0] * 2.0 + X[:, 1]
    skew = np.where(X[:, 0] > 0, 3.0, 0.0)
    y = (base + skew + rng.normal(scale=0.1, size=n)).astype(np.float32)

    feats = compute_sign_residual_baseline_features(X_train=X[:300], y_train=y[:300], X_query=X[300:], seed=42, task="regression", standardize=True)
    mu = feats["signres_mu"].to_numpy()
    baseline_score = feats["signres_baseline_score"].to_numpy()

    assert not np.allclose(mu, baseline_score), "baseline_score must not be a duplicate of mu"
    bias_signal = feats["signres_bias_signal"].to_numpy()
    nonzero_bias = np.abs(bias_signal) > 1e-6
    assert nonzero_bias.any(), "test fixture should induce a nonzero detected bias somewhere"
    assert np.any(np.abs(baseline_score[nonzero_bias] - mu[nonzero_bias]) > 1e-6)
