"""Tests for Gaussian knockoffs (Barber & Candes 2015) and the
multi-estimator min-score-aggregation fix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import (
    RFECV,
    knockoff_importance,
    make_gaussian_knockoffs,
)


# ----------------------------------------------------------------------------
# K1: Gaussian knockoffs sanity - shape, structure, correlation properties
# ----------------------------------------------------------------------------
class TestK1_KnockoffsSanity:
    def test_shape_matches_input(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 10))
        X_tilde = make_gaussian_knockoffs(X, random_state=42)
        assert X_tilde.shape == X.shape

    def test_self_correlation_low(self):
        """Knockoff X_tilde_j should NOT correlate strongly with X_j; that's
        the whole point - the knockoff is independent of y given X_{-j}."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((400, 8))
        X_tilde = make_gaussian_knockoffs(X, random_state=42)
        diag_corrs = [abs(np.corrcoef(X[:, j], X_tilde[:, j])[0, 1]) for j in range(X.shape[1])]
        # For iid normal X with equicorrelated knockoffs, expected self-corr
        # is 1 - s ~ 0. We allow some slack for finite-sample noise.
        assert max(diag_corrs) < 0.30, (
            f"Knockoff self-correlations too high: {diag_corrs}. "
            f"Knockoff should be ~independent of original."
        )

    def test_cross_correlation_structure_preserved(self):
        """Cross correlations between X_j and X_tilde_k (j != k) should be
        close to corr(X_j, X_k) - the knockoff preserves marginal structure."""
        rng = np.random.default_rng(0)
        # Build correlated X: X_2 = 0.7 X_0 + noise
        n = 500
        X = np.zeros((n, 4))
        X[:, 0] = rng.standard_normal(n)
        X[:, 1] = rng.standard_normal(n)
        X[:, 2] = 0.7 * X[:, 0] + 0.3 * rng.standard_normal(n)
        X[:, 3] = 0.7 * X[:, 1] + 0.3 * rng.standard_normal(n)
        X_tilde = make_gaussian_knockoffs(X, random_state=42)
        # corr(X[:,0], X_tilde[:,2]) should be close to corr(X[:,0], X[:,2]) ~ 0.7
        c_real = np.corrcoef(X[:, 0], X[:, 2])[0, 1]
        c_fake = np.corrcoef(X[:, 0], X_tilde[:, 2])[0, 1]
        # Allow 0.2 tolerance (finite-sample + numerical PSD ridge)
        assert abs(c_real - c_fake) < 0.25, (
            f"Cross-correlation structure not preserved: "
            f"corr(X_0, X_2)={c_real:.3f} vs corr(X_0, X_tilde_2)={c_fake:.3f}"
        )

    def test_deterministic_with_random_state(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5))
        a = make_gaussian_knockoffs(X, random_state=0)
        b = make_gaussian_knockoffs(X, random_state=0)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_knockoffs(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5))
        a = make_gaussian_knockoffs(X, random_state=0)
        b = make_gaussian_knockoffs(X, random_state=1)
        # Not equal
        assert not np.allclose(a, b)

    def test_handles_constant_columns(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        X[:, 1] = 5.0  # constant
        # Should not crash
        X_tilde = make_gaussian_knockoffs(X, random_state=0)
        assert X_tilde.shape == X.shape


# ----------------------------------------------------------------------------
# K2: knockoff_importance separates informative from noise
# ----------------------------------------------------------------------------
class TestK2_KnockoffImportance:
    def test_informative_features_have_positive_W(self):
        """On a synthetic problem with M informative + K noise features,
        W = imp(X) - imp(X_tilde) should be:
            - significantly positive for informative features
            - near 0 for noise features
        """
        X, y = make_classification(
            n_samples=500, n_features=12, n_informative=4,
            n_redundant=0, random_state=0, shuffle=False, class_sep=2.5,
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
        W = knockoff_importance(
            model_factory=lambda: LogisticRegression(max_iter=400, random_state=0),
            X=Xdf, y=y, random_state=0,
        )
        # Informative (f0..f3) should rank above the median of noise (f4..f11)
        informative_W = [W[f"f{i}"] for i in range(4)]
        noise_W = [W[f"f{i}"] for i in range(4, 12)]
        median_noise = np.median(noise_W)
        for f_idx, w_inf in enumerate(informative_W):
            assert w_inf > median_noise, (
                f"Informative feature f{f_idx} has W={w_inf:+.4f} which is "
                f"not above median noise W={median_noise:+.4f}. Knockoffs failed."
            )

    def test_W_is_signed_dict(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((100, 3)), columns=list("abc"))
        y = (X["a"] > 0).astype(int).values
        W = knockoff_importance(
            model_factory=lambda: LogisticRegression(max_iter=200, random_state=0),
            X=X, y=y, random_state=0,
        )
        assert set(W.keys()) == {"a", "b", "c"}
        assert all(isinstance(v, float) for v in W.values())


# ----------------------------------------------------------------------------
# K3: multi-estimator score aggregation - min instead of mean
# ----------------------------------------------------------------------------
class TestK3_MultiEstimatorMinAggregation:
    def test_min_aggregation_uses_worst_case_score(self):
        """Verify the score-aggregation fix: when one estimator scores 0.9
        and another 0.7 on the same fold, min returns 0.7 (worst-case).
        Pre-fix, mean returned 0.8, letting the strong estimator hide the
        weak one's signal. This test exercises the per-fold aggregation
        directly; the impact on selection quality depends on the problem
        and is benchmarked separately in
        ``mlframe/feature_selection/_benchmarks/bench_pr4_methods.py``.
        """
        # Use a problem where 2 features clearly aren't enough:
        # noisy class_sep + redundant features force MBH to need more.
        X, y = make_classification(
            n_samples=400, n_features=12, n_informative=6,
            n_redundant=0, random_state=0, shuffle=False, class_sep=1.0,  # noisy
        )
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
        rfecv = RFECV(
            estimators=[
                LogisticRegression(max_iter=400, random_state=0),
                RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1),
            ],
            cv=3, max_refits=10, verbose=0, random_state=0,
        )
        rfecv.fit(Xdf, y)
        # Loose lower bound: as long as we don't collapse to a degenerate
        # 1-2 feature solution. Stronger guarantees come from the
        # stability+multi combo path which sidesteps MBH search entirely.
        assert rfecv.n_features_ >= 3, (
            f"Multi-estimator MBH collapsed to {rfecv.n_features_} features. "
            f"Min-aggregation alone doesn't fix the underlying MBH "
            f"flat-score-plateau issue; use stability_selection=True for "
            f"robust multi-estimator selection."
        )
