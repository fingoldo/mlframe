"""biz_value + unit tests for the Mahalanobis joint density anomaly feature (mrmr_audit_2026-07-20
fe_expansion.md "Multivariate Mahalanobis / Gaussian-copula joint density anomaly score").

Validates ``mahalanobis_density_feature`` (``_mahalanobis_density_fe``): a p-way joint-outlier
score that catches a row far in the FULL correlated joint distribution while every individual
column sits comfortably within its own marginal range.

Contracts pinned
-----------------
* ``TestTypicalRowsScoreLow``: rows drawn from the fitted distribution itself score a modest
  Mahalanobis distance (no false-positive inflation).
* ``TestBizValueJointOutlierInvisibleToMarginals`` (biz_value): a row constructed to be jointly
  atypical (breaks the correlation structure) but marginally ordinary (each column within its own
  1st-99th percentile range) scores a materially higher Mahalanobis distance than typical rows.
* Degenerate inputs (n_fit <= p, non-finite input) return NaN, never raise.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mahalanobis_density_fe import mahalanobis_density_feature


class TestTypicalRowsScoreLow:
    """Rows drawn from the fitted distribution itself must score a modest, non-inflated distance."""

    def test_typical_rows_have_bounded_median_distance(self):
        """Rows drawn from the fitted Gaussian should have a median Mahalanobis^2 on the chi-square(p) scale."""
        rng = np.random.default_rng(0)
        n, p = 3000, 10
        cov = np.eye(p) + 0.3 * (np.ones((p, p)) - np.eye(p))
        X = rng.multivariate_normal(np.zeros(p), cov, size=n)
        d = mahalanobis_density_feature(X)
        assert np.isfinite(d).all()
        # A chi-square(p) reference: median Mahalanobis^2 for a well-specified Gaussian should be
        # near the chi-square(p) median (~p for large p); a generous band avoids over-fitting to
        # exact asymptotics while still catching a badly broken implementation.
        median_d2 = float(np.median(d**2))
        assert 0.4 * p < median_d2 < 2.0 * p, f"median Mahalanobis^2 ({median_d2:.2f}) should be roughly on the chi-square(p={p}) scale"


class TestBizValueJointOutlierInvisibleToMarginals:
    """biz_value: a row that breaks the JOINT correlation structure but stays within each column's
    own marginal range must score a materially higher Mahalanobis distance than typical rows."""

    def test_joint_outlier_scores_higher_than_marginal_range_suggests(self):
        """A row breaking the joint correlation structure but marginally ordinary must score a
        materially higher Mahalanobis distance than the typical-row median."""
        rng = np.random.default_rng(1)
        n, p = 4000, 8
        # Strongly positively correlated columns (a shared latent factor): typical rows have all
        # columns moving TOGETHER.
        latent = rng.standard_normal(n)
        X = np.column_stack([latent + 0.2 * rng.standard_normal(n) for _ in range(p)])

        # A row where HALF the columns are at their high end and half at their low end -- each
        # value individually falls well within the marginal [1st, 99th] percentile range of its own
        # column, but the JOINT combination (half high, half low) never occurs under the strong
        # positive-correlation structure.
        outlier_row = np.array([np.percentile(X[:, j], 90 if j < p // 2 else 10) for j in range(p)])
        for j in range(p):
            lo, hi = np.percentile(X[:, j], [1, 99])
            assert lo < outlier_row[j] < hi, f"column {j}'s outlier value must stay within its own marginal 1st-99th percentile range"

        d_typical = mahalanobis_density_feature(X)
        d_outlier = mahalanobis_density_feature(outlier_row[None, :], X_fit=X)

        median_typical = float(np.median(d_typical))
        assert d_outlier[0] > 3.0 * median_typical, f"joint-outlier Mahalanobis distance ({d_outlier[0]:.4f}) should materially exceed the typical-row median ({median_typical:.4f})"


class TestDegenerateInputsReturnNaN:
    """n_fit <= p and non-finite input must return an all-NaN array, never raise."""

    def test_too_few_fit_rows_returns_nan(self):
        """n_fit <= p (underdetermined covariance) returns all-NaN."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        d = mahalanobis_density_feature(X)
        assert np.isnan(d).all()

    def test_nan_input_returns_nan(self):
        """A NaN anywhere in X (scoring rows) must return all-NaN."""
        rng = np.random.default_rng(2)
        X_fit = rng.standard_normal((500, 3))
        X = np.array([[1.0, np.nan, 3.0]])
        d = mahalanobis_density_feature(X, X_fit=X_fit)
        assert np.isnan(d).all()

    def test_nan_in_fit_data_returns_nan(self):
        """A NaN anywhere in X_fit must return all-NaN, not a poisoned covariance."""
        X_fit = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        X = np.array([[1.0, 2.0]])
        d = mahalanobis_density_feature(X, X_fit=X_fit)
        assert np.isnan(d).all()
