"""Audit 2026-11-20 finding 35-F14: predict_cdf must be correct when the predicted quantile rows are not sorted.

The binary search assumes ascending rows; sorting used to happen only under ``enforce_non_crossing``. A crossing row
``[3, 1, 2]`` at levels ``[.05, .5, .95]`` evaluated at t=2.5 must give the CDF of the sorted quantiles, 0.5.
"""

import numpy as np

from mlframe.training.composite.distributional import CompositeDistributionEstimator


class _Stub(CompositeDistributionEstimator):
    """Skips fitting and serves a fixed, unsorted quantile matrix."""

    def _check_fitted(self):
        """Pretend the model is fitted."""

    def _quantile_matrix(self, X):
        """Return one unsorted quantile row per sample."""
        return np.tile(np.array([3.0, 1.0, 2.0]), (len(X), 1))


def test_cdf_on_unsorted_quantile_row_matches_sorted_row():
    """The CDF at t=2.5 of the crossing row equals the CDF of its sorted version: level .5 is the largest with value <= 2.5."""
    est = _Stub(quantiles=[0.05, 0.5, 0.95])
    est.quantiles_ = np.array([0.05, 0.5, 0.95])
    cdf = est.predict_cdf(np.zeros((2, 1)), [2.5])
    np.testing.assert_allclose(cdf, [[0.5], [0.5]])
