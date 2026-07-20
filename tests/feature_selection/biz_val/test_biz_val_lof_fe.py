"""biz_value + unit tests for the Local Outlier Factor feature (mrmr_audit_2026-07-20
fe_expansion.md "Local Outlier Factor / k-NN local density-ratio feature").

Validates ``lof_scores`` (``_lof_fe``): a LOCAL, non-parametric anomaly score, distinct from a
global elliptical/Mahalanobis anomaly score, that catches a between-cluster gap in a multi-modal
joint distribution.

Contracts pinned
-----------------
* ``TestUniformDensityScoresNearOne``: a single-cluster uniform-density blob gives every row a LOF
  close to 1 (ordinary local density everywhere).
* ``TestBizValueBetweenClusterGap`` (biz_value): on 3 well-separated Gaussian clusters, a point
  planted in the sparse gap BETWEEN clusters gets a materially higher LOF than the cluster
  interior points, even though its Mahalanobis distance to the GLOBAL mean/covariance is
  unremarkable (the global ellipsoid straddles all 3 clusters).
* Degenerate inputs (n<3, non-finite X, k>=n) return NaN / degrade gracefully, never raise.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._lof_fe import lof_scores


class TestUniformDensityScoresNearOne:
    """A single-cluster uniform-density blob must give every interior row a LOF close to 1."""

    def test_gaussian_blob_lof_near_one(self):
        """A single, roughly-uniform-density Gaussian blob has no local density outliers."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2000, 3))
        scores = lof_scores(X, k=20)
        assert np.isfinite(scores).all()
        # Interior rows (excluding the extreme tails) should cluster near LOF=1.
        median_lof = float(np.median(scores))
        assert 0.85 < median_lof < 1.3, f"median LOF on a uniform-density blob should be near 1, got {median_lof:.4f}"


class TestBizValueBetweenClusterGap:
    """biz_value: a point in the sparse gap between 3 clusters gets a high LOF despite an
    unremarkable GLOBAL Mahalanobis distance -- exactly the multi-modal shape LOF catches that a
    global elliptical anomaly score misses."""

    def test_between_cluster_point_scores_higher_than_cluster_interior(self):
        """A point planted in the sparse gap between 3 clusters must score materially higher LOF
        than the cluster-interior median, despite an unremarkable global Mahalanobis distance."""
        rng = np.random.default_rng(1)
        n_per_cluster = 600
        centers = np.array([[-8.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
        clusters = [c + rng.standard_normal((n_per_cluster, 2)) * 0.5 for c in centers]
        X_normal = np.vstack(clusters)

        # A point in the sparse gap between clusters, near the CENTROID of the three cluster
        # centers -- roughly equidistant from all three, in a region no cluster actually occupies.
        gap_point = centers.mean(axis=0, keepdims=True)
        X = np.vstack([X_normal, gap_point])

        scores = lof_scores(X, k=20)
        gap_lof = scores[-1]
        interior_lof_median = float(np.median(scores[:-1]))

        assert gap_lof > 2.0 * interior_lof_median, f"gap point LOF ({gap_lof:.4f}) should be materially higher than cluster-interior median LOF ({interior_lof_median:.4f})"

        # Global Mahalanobis distance check: the gap point's distance to the GLOBAL mean/covariance
        # must NOT already flag it as an outlier by the usual "> 3 sigma"-style threshold -- this is
        # what makes it a genuinely LOCAL (not global) anomaly.
        global_mean = X_normal.mean(axis=0)
        global_cov = np.cov(X_normal, rowvar=False)
        inv_cov = np.linalg.inv(global_cov + 1e-6 * np.eye(2))
        delta = gap_point[0] - global_mean
        mahalanobis_sq = float(delta @ inv_cov @ delta)
        # 2 dof chi-square 99th percentile ~9.21; comfortably inside a "not a global outlier" bound.
        assert mahalanobis_sq < 9.21, f"the gap point should NOT already look like a global outlier (Mahalanobis^2={mahalanobis_sq:.2f}), to isolate LOF's LOCAL detection"


class TestDegenerateInputsReturnNaN:
    """n<3, non-finite input, and k>=n must degrade gracefully rather than raising."""

    def test_two_rows_returns_nan(self):
        """n=2 (below the minimum for a meaningful neighborhood) returns all-NaN."""
        scores = lof_scores(np.array([[1.0, 2.0], [3.0, 4.0]]), k=5)
        assert np.isnan(scores).all()

    def test_nan_input_returns_nan(self):
        """A NaN anywhere in X must return an all-NaN array, not a poisoned distance matrix."""
        X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
        scores = lof_scores(X, k=1)
        assert np.isnan(scores).all()

    def test_k_larger_than_n_minus_one_is_clamped(self):
        """k requesting more neighbors than exist must clamp to n-1, not raise."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((5, 2))
        scores = lof_scores(X, k=100)
        assert scores.shape == (5,)
        assert np.isfinite(scores).all()
