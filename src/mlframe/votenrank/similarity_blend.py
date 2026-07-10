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
from typing import Any, Callable, Optional

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

    Attributes
    ----------
    in_dist_model_, out_dist_model_
        The fitted clones.
    train_embedding_
        Cached training-set embedding, used as the k-NN index at predict time.
    """

    def __init__(
        self,
        in_dist_estimator: Any,
        out_dist_estimator: Any,
        embedding_fn: Callable[[np.ndarray], np.ndarray] = _identity_embedding,
        k: int = 10,
        similarity_scale: float = 1.0,
    ) -> None:
        self.in_dist_estimator = in_dist_estimator
        self.out_dist_estimator = out_dist_estimator
        self.embedding_fn = embedding_fn
        self.k = k
        self.similarity_scale = similarity_scale

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


__all__ = ["SimilarityBlendEnsemble"]
