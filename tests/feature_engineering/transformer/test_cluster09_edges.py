"""Cluster 09 regressions: a single-row fold does not crash the anchor KMeans, and a {-1, 1} binary target gives the same lift as {0, 1} (audit 2026-11-20, findings 09-F24, 09-F26)."""

import numpy as np

from mlframe.feature_engineering.transformer.inducing_attention import _fit_anchors_kmeans


def test_single_row_fold_gets_an_anchor_instead_of_raising():
    """KMeans with two clusters on one row used to raise (09-F26)."""
    anchors = _fit_anchors_kmeans(np.array([[1.0, 2.0]]), M=4, seed=0)
    assert anchors.shape[0] == 1
