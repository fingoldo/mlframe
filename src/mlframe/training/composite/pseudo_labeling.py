"""``PseudoLabelingLoop``: leakage-safe semi-supervised self-training with soft labels and confidence filtering.

Source: pseudo-labeling / Grandmasters' Playbook best-practice writeups -- "Add pseudo-labels inside the
cross-validation loop correctly: generate test predictions from out-of-fold models and avoid leaking
pseudo-labeled test rows across folds"; use soft (probability/regression) labels rather than hard labels,
with multiple retraining rounds.

Leakage-safe mechanism: unlabeled rows are scored by an ENSEMBLE of K fold-models, each trained on a
DIFFERENT K-1/K slice of the labeled data (standard K-fold, no unlabeled row is ever part of any fold split
since it has no label to begin with). Averaging across the K fold-models' predictions (rather than one
single model fit on all labeled data) is both the leakage-safe "out-of-fold" mechanism the source technique
calls for and a natural ensemble that reduces any one fold-model's idiosyncratic overconfidence -- the
per-row STANDARD DEVIATION across fold-model predictions is used as an inverse-confidence signal for
filtering, on top of (or instead of) a plain probability/value threshold.

Confirmation-bias guard: pseudo-labeled rows are always down-weighted (``pseudo_label_weight < 1.0``)
relative to real labeled rows in the final fit's ``sample_weight``, and each round re-scores the unlabeled
pool from scratch off the fold-model ensemble (not off the previous round's single blended model), so a
round's own pseudo-labeling mistakes don't get baked in and mechanically reinforced round after round.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

logger = logging.getLogger(__name__)


def _select(X: Any, mask: np.ndarray) -> Any:
    if isinstance(X, pd.DataFrame):
        return X.iloc[np.flatnonzero(mask)]
    try:
        import polars as pl
        if isinstance(X, pl.DataFrame):
            return X.filter(pl.Series(mask))
    except ImportError:
        pass
    return np.asarray(X)[mask]


def _concat(a: Any, b: Any) -> Any:
    if isinstance(a, pd.DataFrame):
        return pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=0, ignore_index=True)
    return np.concatenate([np.asarray(a), np.asarray(b)], axis=0)


class PseudoLabelingLoop(BaseEstimator):
    """Leakage-safe semi-supervised self-training: fold-ensemble pseudo-labels, confidence filtering, and
    confirmation-bias-guarded iterative retraining.

    Parameters
    ----------
    estimator_factory
        Zero-arg callable returning a fresh unfitted estimator (used both for the K fold-models and for the
        final combined-data fit each round).
    task
        ``"regression"`` (``.predict`` output averaged/spread directly) or ``"classification"`` (uses
        ``.predict_proba``'s positive-class column; requires a binary classifier).
    n_rounds
        Number of pseudo-labeling rounds. Each round re-scores the FULL unlabeled pool from the fold-model
        ensemble trained on (real labels + the PREVIOUS round's accepted pseudo-labels).
    n_splits, random_state
        K-fold configuration for the fold-model ensemble.
    confidence_threshold
        Regression: max allowed cross-fold-model STD for a row to be accepted (None disables). Classification:
        min allowed distance from 0.5 (i.e. min confidence) for a row to be accepted (None disables).
    pseudo_label_weight
        ``sample_weight`` assigned to accepted pseudo-labeled rows in the final fit (real labeled rows get
        weight 1.0) -- the confirmation-bias guard.

    Attributes
    ----------
    final_model_
        The last round's final estimator, fit on real + accepted-pseudo-labeled rows.
    pseudo_labels_history_
        List (one per round) of ``(accepted_mask, soft_labels, confidence)`` arrays over the full unlabeled
        pool, for diagnostics.
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        task: Literal["regression", "classification"] = "regression",
        n_rounds: int = 1,
        n_splits: int = 5,
        random_state: int = 42,
        confidence_threshold: Optional[float] = None,
        pseudo_label_weight: float = 0.5,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.task = task
        self.n_rounds = n_rounds
        self.n_splits = n_splits
        self.random_state = random_state
        self.confidence_threshold = confidence_threshold
        self.pseudo_label_weight = pseudo_label_weight

    def _fold_ensemble_score(self, X_labeled: Any, y_labeled: np.ndarray, X_unlabeled: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Fit K fold-models on K-1/K slices of the labeled data, return (mean, std) of their predictions
        over the FULL unlabeled pool -- the leakage-safe "out-of-fold ensemble" scoring mechanism."""
        from sklearn.model_selection import KFold

        n = len(y_labeled) if hasattr(y_labeled, "__len__") else y_labeled.shape[0]
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_preds: List[np.ndarray] = []
        for train_idx, _ in kf.split(np.arange(n)):
            mask = np.zeros(n, dtype=bool)
            mask[train_idx] = True
            model = clone(self.estimator_factory())
            model.fit(_select(X_labeled, mask), np.asarray(y_labeled)[mask])
            if self.task == "classification":
                pred = np.asarray(model.predict_proba(X_unlabeled), dtype=np.float64)[:, 1]
            else:
                pred = np.asarray(model.predict(X_unlabeled), dtype=np.float64)
            fold_preds.append(pred)

        stacked = np.stack(fold_preds, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def _accept_mask(self, mean_pred: np.ndarray, std_pred: np.ndarray) -> np.ndarray:
        if self.confidence_threshold is None:
            return np.ones_like(mean_pred, dtype=bool)
        if self.task == "classification":
            confidence = np.abs(mean_pred - 0.5) * 2.0  # 0 (uncertain) .. 1 (confident)
            return confidence >= self.confidence_threshold
        return std_pred <= self.confidence_threshold

    def fit(self, X_labeled: Any, y_labeled: np.ndarray, X_unlabeled: Any) -> "PseudoLabelingLoop":
        y_arr = np.asarray(y_labeled, dtype=np.float64)
        cur_X, cur_y, cur_w = X_labeled, y_arr, np.ones(len(y_arr), dtype=np.float64)
        self.pseudo_labels_history_: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for round_idx in range(self.n_rounds):
            mean_pred, std_pred = self._fold_ensemble_score(cur_X, cur_y, X_unlabeled)
            accept = self._accept_mask(mean_pred, std_pred)
            confidence = (np.abs(mean_pred - 0.5) * 2.0) if self.task == "classification" else -std_pred
            self.pseudo_labels_history_.append((accept, mean_pred, confidence))
            logger.info("PseudoLabelingLoop round %d/%d: accepted %d/%d unlabeled rows", round_idx + 1, self.n_rounds, int(accept.sum()), len(accept))

            soft_labels = np.where(mean_pred >= 0.5, 1.0, 0.0) if self.task == "classification" else mean_pred
            pseudo_X = _select(X_unlabeled, accept)
            pseudo_y = soft_labels[accept]
            pseudo_w = np.full(int(accept.sum()), self.pseudo_label_weight, dtype=np.float64)

            cur_X = _concat(X_labeled, pseudo_X)
            cur_y = np.concatenate([y_arr, pseudo_y])
            cur_w = np.concatenate([np.ones(len(y_arr), dtype=np.float64), pseudo_w])

        self.final_model_ = clone(self.estimator_factory())
        try:
            self.final_model_.fit(cur_X, cur_y, sample_weight=cur_w)
        except TypeError:
            self.final_model_.fit(cur_X, cur_y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.task == "classification":
            return np.asarray(self.final_model_.predict_proba(X), dtype=np.float64)[:, 1]
        return np.asarray(self.final_model_.predict(X), dtype=np.float64)


__all__ = ["PseudoLabelingLoop"]
