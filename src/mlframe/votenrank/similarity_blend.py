"""``SimilarityBlendEnsemble``: per-row in-distribution/out-of-distribution blend weight.

Source: Mechanisms of Action Prediction 5th place -- "we decided to discriminate seen and unseen drugs by
metric learning (L2 Softmax -> cosine similarity) and then blended models trained on each validation scheme
based on the similarity." Rather than a fixed blend weight (:func:`confidence_gated_blend`'s binary
step-gate) or a hard hand-off (:class:`mlframe.training.composite.SegmentRoutedEstimator`'s rank-splice),
this computes each query row's k-NN distance to the TRAINING set in an embedding space and converts it to a
continuous similarity-in-[0,1] weight, blending an "in-distribution specialist" and an "out-of-distribution
specialist" model proportionally to it -- a row identical to training data leans fully on the in-distribution
model; a row far from anything seen in training leans fully on the out-of-distribution model.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from ..feature_engineering.transformer._knn_helper import knn_search

logger = logging.getLogger(__name__)


def _identity_embedding(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32)


class SimilarityBlendEnsemble(BaseEstimator, RegressorMixin):
    """Blend an in-distribution and an out-of-distribution specialist model, weighted by each row's
    k-NN similarity to the training set in embedding space.

    Parameters
    ----------
    in_dist_estimator, out_dist_estimator
        sklearn-compatible estimator prototypes, cloned and fit independently at fit time. Typically the
        in-distribution model is trained/validated on rows resembling the bulk of production traffic, and
        the out-of-distribution model on a scheme robust to novel entities (the caller decides what
        "in-distribution" vs. "out-of-distribution" training means -- this class only handles the blend).
    embedding_fn
        ``callable(X) -> (n, d) ndarray``. Defaults to the identity (raw ``X`` as the similarity space) --
        pass a fitted embedding/projection for a metric-learned space (matching the source technique's
        L2-softmax embedding) when raw feature-space distance isn't the right similarity notion.
    k
        Number of nearest training rows to average distance over per query row.
    similarity_scale
        Distance-to-similarity conversion: ``similarity = exp(-mean_knn_distance / similarity_scale)``.
        Larger values make the blend weight less sensitive to distance (flatter falloff); tune to the
        embedding space's natural distance scale (e.g. the training set's own median pairwise distance).
    region_estimators
        Opt-in N-specialist mode. ``None`` (default) leaves ``fit``/``predict`` on the original 2-specialist
        path, bit-identical to before this parameter existed. When set to a list of N sklearn-compatible
        estimator prototypes, use :meth:`fit_multi_region`/:meth:`predict_multi_region` instead: each
        specialist is fit on its own region's training rows, and blended per query row by a softmax-normalized
        similarity to each region's own training-set distribution (generalizing the binary in/out-of-region
        gate above to N regions with a soft, non-binary weight vector that sums to 1).

    Attributes
    ----------
    in_dist_model_, out_dist_model_
        The fitted clones (2-specialist path).
    train_embedding_
        Cached training-set embedding, used as the k-NN index at predict time (2-specialist path).
    region_models_, region_embeddings_
        Per-region fitted clones and cached training-set embeddings (N-specialist path).
    """

    def __init__(
        self,
        in_dist_estimator: Any,
        out_dist_estimator: Any,
        embedding_fn: Callable[[np.ndarray], np.ndarray] = _identity_embedding,
        k: int = 10,
        similarity_scale: float = 1.0,
        region_estimators: Optional[Sequence[Any]] = None,
    ) -> None:
        self.in_dist_estimator = in_dist_estimator
        self.out_dist_estimator = out_dist_estimator
        self.embedding_fn = embedding_fn
        self.k = k
        self.similarity_scale = similarity_scale
        self.region_estimators = region_estimators

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None) -> "SimilarityBlendEnsemble":
        y_arr = np.asarray(y, dtype=np.float64)
        self.in_dist_model_ = clone(self.in_dist_estimator)
        self.out_dist_model_ = clone(self.out_dist_estimator)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.in_dist_model_.fit(X, y_arr, **fit_kwargs)
        self.out_dist_model_.fit(X, y_arr, **fit_kwargs)
        self.train_embedding_ = self.embedding_fn(np.asarray(X, dtype=np.float32))
        return self

    def similarity_weight(self, X: Any) -> np.ndarray:
        """Return each query row's in-distribution blend weight in ``[0, 1]``."""
        query_embedding = self.embedding_fn(np.asarray(X, dtype=np.float32))
        k_eff = min(self.k, self.train_embedding_.shape[0])
        dists, _ = knn_search(self.train_embedding_, query_embedding, k_eff)
        mean_dist = dists.mean(axis=1)
        return np.asarray(np.exp(-mean_dist / self.similarity_scale), dtype=np.float64)

    def predict(self, X: Any) -> np.ndarray:
        w = self.similarity_weight(X)
        in_pred = np.asarray(self.in_dist_model_.predict(X), dtype=np.float64)
        out_pred = np.asarray(self.out_dist_model_.predict(X), dtype=np.float64)
        return w * in_pred + (1.0 - w) * out_pred

    def fit_multi_region(
        self,
        region_X: Sequence[Any],
        region_y: Sequence[Any],
        sample_weight: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> "SimilarityBlendEnsemble":
        """Fit N region-specific specialists, one per entry of ``region_X``/``region_y``.

        Requires ``region_estimators`` to have been set in ``__init__`` to a list of N estimator prototypes,
        one per region, cloned and fit independently here -- mirroring ``fit``'s in/out-of-distribution split
        but generalized to N caller-defined regions instead of exactly 2.
        """
        if self.region_estimators is None:
            raise ValueError("region_estimators must be set in __init__ to use fit_multi_region (N-specialist mode)")
        if len(region_X) != len(self.region_estimators) or len(region_y) != len(self.region_estimators):
            raise ValueError(f"expected {len(self.region_estimators)} region_X/region_y entries (one per region_estimators), got {len(region_X)}/{len(region_y)}")
        region_models: List[Any] = []
        region_embeddings: List[np.ndarray] = []
        for i, (Xr, yr) in enumerate(zip(region_X, region_y)):
            yr_arr = np.asarray(yr, dtype=np.float64)
            model = clone(self.region_estimators[i])
            sw_i = sample_weight[i] if sample_weight is not None else None
            fit_kwargs = {"sample_weight": sw_i} if sw_i is not None else {}
            model.fit(Xr, yr_arr, **fit_kwargs)
            region_models.append(model)
            region_embeddings.append(self.embedding_fn(np.asarray(Xr, dtype=np.float32)))
        self.region_models_ = region_models
        self.region_embeddings_ = region_embeddings
        return self

    def region_similarity_weights(self, X: Any) -> np.ndarray:
        """Return each query row's ``(n, n_regions)`` blend weight matrix, rows summing to 1.

        Each column is the row's k-NN similarity to that region's own training-set distribution
        (same distance-to-similarity conversion as :meth:`similarity_weight`), softmax-normalized across
        regions -- a row deep inside region j's training distribution and far from every other region's
        training data leans almost fully on specialist j, while a row equidistant from several regions
        blends them roughly evenly, instead of the hard binary split of the 2-specialist path.
        """
        query_embedding = self.embedding_fn(np.asarray(X, dtype=np.float32))
        n_regions = len(self.region_embeddings_)
        sims = np.empty((query_embedding.shape[0], n_regions), dtype=np.float64)
        for i, train_emb in enumerate(self.region_embeddings_):
            k_eff = min(self.k, train_emb.shape[0])
            dists, _ = knn_search(train_emb, query_embedding, k_eff)
            mean_dist = dists.mean(axis=1)
            sims[:, i] = np.exp(-mean_dist / self.similarity_scale)
        row_sums = sims.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        return np.asarray(sims / row_sums, dtype=np.float64)

    def predict_multi_region(self, X: Any) -> np.ndarray:
        """Blend the N region specialists' predictions per row, weighted by :meth:`region_similarity_weights`."""
        weights = self.region_similarity_weights(X)
        preds = np.column_stack([np.asarray(model.predict(X), dtype=np.float64) for model in self.region_models_])
        return np.asarray(np.sum(weights * preds, axis=1), dtype=np.float64)


__all__ = ["SimilarityBlendEnsemble"]
