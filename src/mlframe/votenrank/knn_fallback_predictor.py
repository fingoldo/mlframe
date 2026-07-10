"""``KNNFallbackPredictor``: fit/predict kNN target-average, meant as a low-confidence fallback in a blend.

Source: 3rd_mechanisms-of-action-moa-prediction.md -- "average of the five most similar train samples" via
cosine similarity used to predict private-set values with weak/no direct signal from the primary model.

mlframe already has kNN target-mean logic for FEATURE ENGINEERING
(``feature_engineering.transformer.neighbor_aggregate_features``, fold-driven, OOF-safe) and a generic
confidence-gated blend plumbing (``votenrank.confidence_gated_blend.confidence_gated_blend``, metric-agnostic
weighted gate between an ensemble and an auxiliary prediction) -- but nothing exposes a standalone
``fit(X, y)``/``predict(X_query) -> (pred, confidence)`` object in the shape ``confidence_gated_blend``
expects for its ``auxiliary_pred``/``auxiliary_confidence`` inputs. This fills that gap, reusing the existing
``knn_search`` primitive (HNSW/sklearn dispatch) rather than reimplementing nearest-neighbor search.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from mlframe.feature_engineering.transformer._knn_helper import knn_search


class KNNFallbackPredictor:
    """A trained kNN target-average predictor with a confidence score, for use as a blend fallback.

    Parameters
    ----------
    k
        Number of nearest labeled neighbors to average.
    metric
        Distance metric passed to ``knn_search`` (``"l2"`` or ``"cosine"``).
    """

    def __init__(self, k: int = 5, metric: str = "l2") -> None:
        self.k = k
        self.metric = metric
        self._X_train: np.ndarray = np.empty((0, 0))
        self._y_train: np.ndarray = np.empty(0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNFallbackPredictor":
        self._X_train = np.asarray(X, dtype=np.float32)
        self._y_train = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, X_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict via k-nearest-neighbor target average, with an inverse-distance confidence score.

        Returns
        -------
        tuple
            ``(pred, confidence)`` -- ``pred`` is the mean target of the ``k`` nearest training rows to each
            query row; ``confidence`` is ``1 / (1 + mean_distance)`` (in ``(0, 1]``, HIGH when the nearest
            neighbors are close -- a genuinely similar match -- LOW when even the closest neighbors are far,
            i.e. the query row is in a sparse/unfamiliar region of feature space where the fallback itself
            shouldn't be trusted much either).
        """
        X_query_arr = np.asarray(X_query, dtype=np.float32)
        dists, ids = knn_search(self._X_train, X_query_arr, self.k, metric=self.metric)
        neighbor_targets = self._y_train[ids]
        pred = np.asarray(np.mean(neighbor_targets, axis=1))
        mean_dist = np.asarray(np.mean(dists, axis=1))
        confidence = np.asarray(1.0 / (1.0 + mean_dist))
        return pred, confidence


__all__ = ["KNNFallbackPredictor"]
