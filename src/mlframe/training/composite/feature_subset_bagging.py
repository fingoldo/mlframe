"""``FeatureSubsetBaggingEnsemble``: correlation-cluster-aware feature-subset bagging.

Source: 4th_home-credit-default-risk.md -- "train diverse models on subset of features and on overall big
feature set ... trained over 200 models on different parts with different models." A comment in the same
thread reports plain ``sklearn.utils.random.sample_without_replacement`` feature bagging gave NO gain --
purely random feature subsets tend to either duplicate the same dominant correlated-feature-cluster signal
across subsets (no real diversity) or split a cluster's signal so thin no single subset captures it well.
This implementation samples PROPORTIONALLY ACROSS CORRELATION CLUSTERS instead (hierarchical clustering on
``1 - |corr|``, at least one feature per cluster per subset when the subset size allows), so each subset
genuinely covers a different slice of the feature space's independent signal directions rather than an
arbitrary random draw.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score


def correlation_cluster_feature_subsets(
    X: pd.DataFrame,
    n_subsets: int,
    subset_size: int,
    n_clusters: int = 10,
    random_state: int = 42,
) -> List[List[str]]:
    """Sample ``n_subsets`` feature subsets, each drawing proportionally across correlation clusters.

    Parameters
    ----------
    X
        Feature frame (numeric columns only are clustered).
    n_subsets
        Number of feature subsets to generate.
    subset_size
        Number of features per subset.
    n_clusters
        Number of correlation clusters (hierarchical clustering on ``1 - |corr|``); capped at
        ``min(n_clusters, n_features)``.
    random_state
        Seed.

    Returns
    -------
    list of list of str
        ``n_subsets`` feature-name lists, each length ``subset_size`` (or fewer if ``subset_size`` exceeds
        the available feature count).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    cols = list(X.select_dtypes(include=[np.number]).columns)
    n_features = len(cols)
    k = min(n_clusters, n_features)

    corr = X[cols].corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry (float round-trip can drift by ~1e-16)
    condensed = squareform(dist, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    cluster_labels = fcluster(linkage_matrix, t=k, criterion="maxclust")

    clusters: dict = {}
    for col, label in zip(cols, cluster_labels):
        clusters.setdefault(label, []).append(col)
    cluster_lists = list(clusters.values())

    rng = np.random.default_rng(random_state)
    subsets = []
    for _ in range(n_subsets):
        subset: List[str] = []
        # round-robin across clusters until subset_size is reached, sampling one feature per cluster per pass.
        # visitation ORDER is reshuffled per subset (not just the within-cluster feature order) -- otherwise,
        # whenever subset_size < n_clusters, every subset would deterministically draw from the same fixed
        # first-N clusters (only WHICH feature inside each cluster would vary), defeating the whole point of
        # covering different slices of the feature space across subsets.
        cluster_order = rng.permutation(len(cluster_lists))
        cluster_pools = [list(rng.permutation(cluster_lists[i])) for i in cluster_order]
        while len(subset) < min(subset_size, n_features):
            for pool in cluster_pools:
                if pool and len(subset) < subset_size:
                    subset.append(pool.pop())
            if all(not pool for pool in cluster_pools):
                break
        subsets.append(subset)
    return subsets


class FeatureSubsetBaggingEnsemble(BaseEstimator, RegressorMixin):
    """Bag several models, each trained on a correlation-cluster-diverse feature subset.

    Parameters
    ----------
    estimator_factory
        Zero-arg callable returning a fresh unfitted regressor.
    n_subsets
        Number of feature-subset models.
    subset_size
        Features per subset.
    n_clusters
        Correlation clusters to sample proportionally across (see :func:`correlation_cluster_feature_subsets`).
    random_state
        Seed.
    aggregation
        ``"mean"`` (default, prior behavior -- unweighted average of sub-model predictions) or ``"weighted"``
        (opt-in): each sub-model is weighted by its own OOF R2 on the training fold (via
        ``sklearn.model_selection.cross_val_predict``, so the weight is honest -- never scored on rows the
        sub-model itself was fit on), clipped at 0 (a worse-than-mean-baseline subset gets zero say, not a
        negative vote). Wins when subsets differ genuinely in informativeness -- e.g. a feature-subset scheme
        that can draw an almost-pure-noise cluster into some subsets while others land mostly-signal clusters
        -- because plain averaging lets the noisy sub-models drag the pooled prediction down.

        Not built on :func:`mlframe.models.ensembling.selection.rank_average_blend`: that primitive
        rank-transforms scores across rows first (AUC/classification-oriented, scale-invariant by design),
        which would discard the regressor's actual output scale here. A direct R2-weighted mean of the raw
        regression predictions is the correct analog for this contract, so the OOF-weight computation is a
        thin dedicated wrapper around ``cross_val_predict`` instead.
    weighted_cv
        Number of folds for the per-subset OOF weighting when ``aggregation="weighted"`` (default 3, ignored
        otherwise).
    """

    weights_: Optional[np.ndarray]

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        n_subsets: int = 5,
        subset_size: int = 10,
        n_clusters: int = 10,
        random_state: int = 42,
        aggregation: str = "mean",
        weighted_cv: int = 3,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.aggregation = aggregation
        self.weighted_cv = weighted_cv

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FeatureSubsetBaggingEnsemble":
        if self.aggregation not in ("mean", "weighted"):
            raise ValueError(f"FeatureSubsetBaggingEnsemble: aggregation must be 'mean' or 'weighted', got {self.aggregation!r}.")
        self.feature_subsets_ = correlation_cluster_feature_subsets(X, self.n_subsets, self.subset_size, self.n_clusters, self.random_state)
        self.estimators_ = []
        y_arr = np.asarray(y)
        weights: List[float] = []
        for subset in self.feature_subsets_:
            model = self.estimator_factory()
            model.fit(X[subset], y_arr)
            self.estimators_.append(model)
            if self.aggregation == "weighted":
                n_splits = min(self.weighted_cv, len(X))
                if n_splits < 2:
                    weights.append(1.0)
                else:
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                    oof_pred = cross_val_predict(self.estimator_factory(), X[subset], y_arr, cv=cv)
                    weights.append(max(0.0, float(r2_score(y_arr, oof_pred))))
        if self.aggregation == "weighted":
            total = sum(weights)
            # a fully-degenerate weight set (all subsets score <=0 OOF) falls back to uniform rather than
            # producing a NaN blend -- every sub-model still contributes, just as the prior "mean" behavior did.
            self.weights_ = np.full(len(weights), 1.0 / len(weights)) if total <= 0.0 else np.asarray(weights) / total
        else:
            self.weights_ = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack([model.predict(X[subset]) for model, subset in zip(self.estimators_, self.feature_subsets_)], axis=0)
        if self.aggregation == "weighted" and self.weights_ is not None:
            return np.asarray(np.average(preds, axis=0, weights=self.weights_))
        return np.asarray(np.mean(preds, axis=0))


__all__ = ["FeatureSubsetBaggingEnsemble", "correlation_cluster_feature_subsets"]
