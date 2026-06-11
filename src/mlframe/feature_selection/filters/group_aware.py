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

from mlframe.utils.misc import rng_hygienic_fit

logger = logging.getLogger(__name__)


def _numeric_codes_frame(X: "pd.DataFrame") -> "pd.DataFrame":
    """Return a float DataFrame where non-numeric columns are replaced by their
    integer factor codes, so correlation / SU clustering never float-coerces a
    raw categorical / object / string column (e.g. a ``"NA"`` level) and crashes.

    Redundancy clustering is an unsupervised same-row operation; per-column
    ``factorize`` (NaN -> code -1) is a valid integer encoding that preserves the
    original column count and order, so cluster indices still map back to the
    caller's feature positions. Numeric columns pass through unchanged.
    """
    # Iterate POSITIONALLY (``.iloc[:, j]``), not by label: duplicate column names (common after FE expansion -- repeated
    # lags, one-hot level collisions) make ``X[label]`` return a DataFrame, whose ``.dtype`` access raises. Positional
    # access always yields a Series, and we rebuild from a list so duplicate labels are preserved in original order.
    series_list = []
    for j in range(X.shape[1]):
        s = X.iloc[:, j]
        if pd.api.types.is_numeric_dtype(s.dtype):
            series_list.append(np.asarray(s.to_numpy()))
        else:
            codes, _ = pd.factorize(s, use_na_sentinel=True)
            series_list.append(np.asarray(codes))
    out = pd.DataFrame(np.column_stack(series_list) if series_list else np.empty((len(X), 0)), index=X.index)
    out.columns = list(X.columns)
    return out


def _su_redundancy_matrix(X, nbins: int = 10) -> np.ndarray:
    """Pairwise Symmetric-Uncertainty redundancy matrix in [0, 1] (diagonal 0).

    Unlike Pearson/Spearman (which only see monotone dependence), SU captures arbitrary -- including
    non-linear / non-monotone -- redundancy between two columns: two features that are deterministic but
    non-monotone functions of each other score ~1 here but ~0 under Pearson. Each column is quantile-binned
    to ``nbins`` codes (continuous) or used as-is when already low-cardinality, then SU = 2*I/(H_a+H_b) is
    computed per pair from the joint bincount. O(p^2) over the (already cluster-bounded) feature set.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    arr = _numeric_codes_frame(X).to_numpy()
    n, p = arr.shape
    codes = np.empty((n, p), dtype=np.int64)
    ncats = np.empty(p, dtype=np.int64)
    for j in range(p):
        col = arr[:, j]
        col = np.asarray(col, dtype=np.float64)
        finite = np.isfinite(col)
        uniq = np.unique(col[finite]) if finite.any() else np.array([0.0])
        if uniq.shape[0] <= nbins:
            # Low-cardinality / already-discrete: map values to dense codes.
            lookup = {v: i for i, v in enumerate(uniq)}
            c = np.array([lookup.get(v, len(lookup)) for v in col], dtype=np.int64)
            ncats[j] = len(lookup) + (0 if finite.all() else 1)
        else:
            qs = np.quantile(col[finite], np.linspace(0, 1, nbins + 1)[1:-1])
            c = np.digitize(col, qs).astype(np.int64)
            c[~finite] = nbins  # NaN sentinel bin
            ncats[j] = nbins + (0 if finite.all() else 1)
        codes[:, j] = c

    def _entropy(counts):
        tot = counts.sum()
        if tot <= 0:
            return 0.0
        pr = counts[counts > 0] / tot
        return float(-(pr * np.log(pr)).sum())

    h = np.array([_entropy(np.bincount(codes[:, j], minlength=int(ncats[j]))) for j in range(p)])
    su = np.zeros((p, p), dtype=np.float64)
    for a in range(p):
        for b in range(a + 1, p):
            joint = np.bincount(codes[:, a] * int(ncats[b]) + codes[:, b], minlength=int(ncats[a] * ncats[b]))
            h_ab = _entropy(joint)
            mi_ab = h[a] + h[b] - h_ab
            denom = h[a] + h[b]
            val = 0.0 if denom <= 1e-12 else max(0.0, 2.0 * mi_ab / denom)
            su[a, b] = su[b, a] = val
    return su


def _redundancy_matrix(X, method: str) -> np.ndarray:
    """Abs redundancy matrix (diagonal 0) for the given method. ``'su'`` uses Symmetric Uncertainty (captures non-linear redundancy); everything else delegates to ``pandas.DataFrame.corr``."""
    if method == "su":
        return _su_redundancy_matrix(X)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    # Factor-code non-numeric columns first: ``DataFrame.corr`` float-coerces every
    # column and would crash on a raw categorical / string level (e.g. ``"NA"``).
    corr = np.array(_numeric_codes_frame(X).corr(method=method).abs().to_numpy(), copy=True)
    np.fill_diagonal(corr, 0.0)
    return corr


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
    method : {"pearson", "spearman", "kendall", "su"}
        Redundancy measure. The corr methods go through ``pandas.DataFrame.corr``; ``"su"`` uses Symmetric
        Uncertainty (captures non-linear / non-monotone redundancy the corr methods miss).

    Returns
    -------
    cluster_id : ndarray of shape (n_features,)
        Integer cluster labels in [0, n_clusters - 1].
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    corr = _redundancy_matrix(X, method)
    n = corr.shape[0]

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
    corr = _redundancy_matrix(X, method)
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

    @staticmethod
    def _inner_support_indices(inner, columns):
        """Integer column indices the inner selector kept, normalising across
        selector conventions so GroupAwareMRMR wraps any of them:
          * ``support_`` -- boolean mask OR index array (sklearn RFECV, mRMR);
          * ``get_support()`` -- sklearn SelectorMixin;
          * ``accepted`` -- list of kept column NAMES (BorutaShap).
        ``columns`` is the ordered column names the inner was fit on (X or the
        medoid subset), used to map ``accepted`` names back to positions.
        """
        sup = getattr(inner, "support_", None)
        if sup is not None:
            sup = np.asarray(sup)
            return np.where(sup)[0] if sup.dtype == bool else sup.astype(np.int64)
        if hasattr(inner, "get_support"):
            try:
                sup = np.asarray(inner.get_support())
                return np.where(sup)[0] if sup.dtype == bool else sup.astype(np.int64)
            except Exception:
                pass
        accepted = getattr(inner, "accepted", None)  # BorutaShap: kept col names
        if accepted is not None:
            pos = {str(c): i for i, c in enumerate(columns)}
            return np.array([pos[str(c)] for c in accepted if str(c) in pos], dtype=np.int64)
        raise AttributeError(
            f"{type(inner).__name__} exposes no support_/get_support()/accepted; "
            f"GroupAwareMRMR cannot map its selection back to clusters."
        )

    @rng_hygienic_fit
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
            self.support_ = self._inner_support_indices(inner, list(X.columns))
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

        # Map the inner selector's kept medoids back to clusters via the
        # convention-agnostic helper (support_ index/mask, get_support(), or
        # BorutaShap's ``accepted`` names). 2026-06-03 (audit integration-
        # defaults-3): lets GroupAwareMRMR wrap ANY wrapper selector
        # (RFECV / BorutaShap) -- a MEASURED ~1.4-3x wall-clock win with no OOS
        # loss -- by running the wrapper on the cluster medoids instead of every
        # redundant column.
        sel_idx = self._inner_support_indices(inner, list(X_medoids.columns))
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

    @property
    def accepted(self):
        """BorutaShap report contract: the training suite reads ``accepted``
        (kept column NAMES) for BorutaShap diagnostics. Return the EXPANDED
        selection (cluster members of accepted medoids) -- i.e. the same set
        ``support_`` / ``transform`` / ``get_feature_names_out`` expose -- so the
        report agrees with what actually feeds training, NOT the inner's
        medoid-only accepted list. Falls through (AttributeError -> __getattr__ ->
        inner) when the wrapper isn't fitted or the inner has no ``accepted``.
        """
        inner = self.__dict__.get("estimator_")
        if inner is not None and hasattr(inner, "accepted") and "support_" in self.__dict__:
            return list(self.get_feature_names_out())
        raise AttributeError("accepted")

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
