"""Group-aware mRMR via correlation-based pre-clustering.

Vanilla mRMR with high-correlation feature sets (one-hot expansion, repeated lags, calibration variants of the same sensor) selects ONE
representative and discards the rest. Often the operator wants the **group**, not just one member -- either to display them together in
a downstream report or to feed them into an ensemble that benefits from the redundancy.

Two-step approach:

1. ``cluster_features_by_correlation(X, threshold=0.9, ...)`` -- greedy clustering: every pair with ``|corr| > threshold`` ends in the
   same cluster (single-linkage on the correlation graph). Returns a ``cluster_id`` per feature.
2. ``GroupAwareMRMR(estimator, ...).fit(X, y)`` -- runs mRMR on the per-cluster medoids (the feature with highest mean abs-corr to its
   cluster mates), then expands the support to all members of any selected cluster. ``cluster_assignments_`` and ``selected_clusters_``
   are exposed for inspection.

For users with **explicit** group structure (one-hot expansions where the operator knows ``group_name -> column_list``), prefer
``RFECV(feature_groups=...)`` -- it has a more thorough all-or-nothing voting protocol. This module is for **discovered** groups via
correlation when the operator does not know the structure upfront.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone

logger = logging.getLogger(__name__)


def cluster_features_by_correlation(
    X,
    threshold: float = 0.9,
    method: str = "spearman",
) -> np.ndarray:
    """Greedy single-linkage clustering on the correlation graph.

    Two features land in the same cluster iff there's a chain of pairwise ``|corr| > threshold`` connections between them.

    Parameters
    ----------
    X : array or DataFrame, shape (n_samples, n_features)
    threshold : float, default 0.9
    method : {"pearson", "spearman", "kendall"}
        Correlation method passed to ``pandas.DataFrame.corr``.

    Returns
    -------
    cluster_id : ndarray of shape (n_features,)
        Integer cluster labels in [0, n_clusters - 1].
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    # pandas 2.x `.corr().abs().to_numpy()` may return a read-only zero-copy
    # view of the underlying block (Arrow-backed frames especially).
    # ``np.fill_diagonal`` writes in-place so we need a writable copy.
    corr = np.array(X.corr(method=method).abs().to_numpy(), copy=True)
    n = corr.shape[0]
    np.fill_diagonal(corr, 0.0)

    cluster_id = np.arange(n)  # union-find roots

    def _find(i: int) -> int:
        while cluster_id[i] != i:
            cluster_id[i] = cluster_id[cluster_id[i]]
            i = cluster_id[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            cluster_id[max(ri, rj)] = min(ri, rj)

    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] > threshold:
                _union(i, j)

    # Compress + relabel.
    roots = np.array([_find(i) for i in range(n)])
    _, compact = np.unique(roots, return_inverse=True)
    return compact


def _cluster_medoids(
    X,
    cluster_id: np.ndarray,
    method: str = "spearman",
) -> list[int]:
    """For each cluster, pick the column with highest mean abs-corr to its cluster mates (the medoid). Singletons return their only member."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    corr = X.corr(method=method).abs().to_numpy()
    n_clusters = int(cluster_id.max()) + 1

    medoids = []
    for c in range(n_clusters):
        members = np.where(cluster_id == c)[0]
        if len(members) == 1:
            medoids.append(int(members[0]))
            continue
        sub = corr[np.ix_(members, members)]
        # mean abs-corr to siblings (exclude self via zero diagonal).
        np.fill_diagonal(sub, 0.0)
        scores = sub.mean(axis=1)
        medoids.append(int(members[np.argmax(scores)]))
    return medoids


class GroupAwareMRMR(BaseEstimator, TransformerMixin):
    """Wraps an mRMR-family estimator with correlation pre-clustering.

    .fit fits the inner estimator on cluster medoids; .transform / .support_ expand to all cluster members of any selected medoid.

    Attributes after fit:
    * ``cluster_assignments_`` -- per-original-feature cluster id.
    * ``cluster_medoid_indices_`` -- per-cluster representative index (in original-feature space).
    * ``selected_clusters_`` -- ids of clusters whose medoid mRMR kept.
    * ``support_`` -- expanded set of all original-feature indices belonging to any selected cluster.
    """
    def __init__(
        self,
        estimator,
        corr_threshold: float = 0.9,
        corr_method: str = "spearman",
        expand: bool = True,
        min_reduction: float = 0.05,
    ):
        self.estimator = estimator
        self.corr_threshold = corr_threshold
        self.corr_method = corr_method
        self.expand = expand
        # 2026-06-03 (audit integration-defaults-3): SAFETY/USEFULNESS guard for
        # default-ON use. The cluster-medoid reduction is a measured win
        # (~1.4-1.9x, no OOS loss) ONLY when genuine correlated redundancy
        # exists; on near-uncorrelated data it would reduce nothing yet still run
        # the wrapper on a same-size medoid set. When the clustering eliminates
        # fewer than ``min_reduction`` of the features, we BYPASS the medoid path
        # and fit the inner selector on the full X -> identical selection to a
        # bare wrapper, no wasted wrapper work. Broad validation
        # (bench_cross_selector_diverse) showed AUC delta in [-0.0004, +0.0081]
        # across synthetic + real datasets -> safe to default ON.
        self.min_reduction = min_reduction

    def fit(self, X, y, **fit_params):
        # **fit_params (e.g. ``groups`` for a GroupKFold cv, ``sample_weight``)
        # are row-aligned, so they pass straight through to the inner selector
        # whether it fits on the medoid subset or the full X (same rows).
        is_df = isinstance(X, pd.DataFrame)
        if not is_df:
            X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        self.cluster_assignments_ = cluster_features_by_correlation(
            X, threshold=self.corr_threshold, method=self.corr_method,
        )
        self.cluster_medoid_indices_ = _cluster_medoids(
            X, self.cluster_assignments_, method=self.corr_method,
        )
        n_clusters = len(self.cluster_medoid_indices_)
        n_feat = X.shape[1]
        reduction = (n_feat - n_clusters) / max(n_feat, 1)
        self.reduction_ = float(reduction)
        self.reduced_ = bool(reduction >= float(self.min_reduction))
        logger.info(
            "GroupAwareMRMR: %d original features -> %d cluster medoids "
            "(corr_threshold=%.2f, reduction=%.1f%%, applied=%s).",
            n_feat, n_clusters, self.corr_threshold, 100.0 * reduction,
            self.reduced_,
        )

        # Guard: when the reduction is below the threshold, run the inner
        # selector on the FULL feature set (no medoid bypass) so the result is
        # identical to a bare wrapper and no wrapper work is wasted clustering.
        if not self.reduced_:
            inner = clone(self.estimator)
            inner.fit(X, y, **fit_params)
            self.estimator_ = inner
            _sup_full = np.asarray(inner.support_)
            self.support_ = (
                np.where(_sup_full)[0] if _sup_full.dtype == bool
                else _sup_full.astype(np.int64)
            )
            self.selected_clusters_ = sorted(
                set(int(self.cluster_assignments_[i]) for i in self.support_)
            )
            self.n_features_ = len(self.support_)
            self.n_features_in_ = n_feat
            if is_df:
                self.feature_names_in_ = list(X.columns)
            return self

        # Fit inner estimator on the medoid subset only.
        X_medoids = X.iloc[:, self.cluster_medoid_indices_]
        inner = clone(self.estimator)
        inner.fit(X_medoids, y, **fit_params)
        self.estimator_ = inner

        # Map the inner selector's kept medoids back to clusters. Normalise
        # ``support_`` to integer indices into X_medoids: the mRMR family exposes
        # an index array, while sklearn-style wrappers (RFECV) expose a boolean
        # mask. 2026-06-03 (audit integration-defaults-3): this generalisation
        # lets GroupAwareMRMR wrap ANY wrapper selector (RFECV / BorutaShap),
        # which on wide correlated data is a MEASURED ~3x wall-clock win with no
        # OOS loss (bench_cross_selector_cluster_reduction) -- the wrapper runs
        # on the cluster medoids instead of every redundant column.
        _sup = np.asarray(inner.support_)
        sel_idx = np.where(_sup)[0] if _sup.dtype == bool else _sup.astype(np.int64)
        medoid_cluster_ids = self.cluster_assignments_[self.cluster_medoid_indices_]
        self.selected_clusters_ = sorted(set(int(medoid_cluster_ids[int(i)]) for i in sel_idx))

        if self.expand:
            self.support_ = np.array(sorted([
                idx for idx in range(X.shape[1])
                if self.cluster_assignments_[idx] in self.selected_clusters_
            ]), dtype=np.int64)
        else:
            # Just the medoids of selected clusters.
            self.support_ = np.array(sorted([
                self.cluster_medoid_indices_[c] for c in self.selected_clusters_
            ]), dtype=np.int64)

        self.n_features_ = len(self.support_)
        self.n_features_in_ = X.shape[1]
        if is_df:
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X, y=None):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]
        return X[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        """Selected feature names (sklearn transformer contract) so this is a
        faithful drop-in inside the training-suite pre-pipeline. ``support_`` is
        an integer index array into ``feature_names_in_``."""
        names = getattr(self, "feature_names_in_", None)
        if names is not None:
            return np.asarray([names[int(i)] for i in self.support_], dtype=object)
        if input_features is not None:
            return np.asarray([input_features[int(i)] for i in self.support_], dtype=object)
        return np.asarray([f"f_{int(i)}" for i in self.support_], dtype=object)

    def __getattr__(self, name):
        """Transparently expose the fitted inner selector's attributes (e.g. the
        training-suite's ``_selector_kind`` / ``_mlframe_use_sample_weights_in_fs_``
        markers, RFECV's ``cv_results_`` etc.) so this wrapper is a faithful
        drop-in. Only consulted for attributes GroupAwareMRMR does NOT define
        itself (support_, transform, get_feature_names_out, etc. stay the
        wrapper's expanded versions). Guarded against pre-fit recursion.
        """
        if name in ("estimator_", "estimator", "__setstate__", "__getstate__"):
            raise AttributeError(name)
        inner = self.__dict__.get("estimator_")
        if inner is not None and hasattr(inner, name):
            return getattr(inner, name)
        raise AttributeError(
            f"{type(self).__name__!r} object (and its inner estimator) has no "
            f"attribute {name!r}"
        )
