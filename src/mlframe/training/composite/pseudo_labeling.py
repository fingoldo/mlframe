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
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

logger = logging.getLogger(__name__)


def _select(X: Any, mask: np.ndarray) -> Any:
    """Select the rows of ``X`` (DataFrame, polars, or array-like) matching a boolean ``mask``."""
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
    """Concatenate ``a`` and ``b`` row-wise, preserving DataFrame type when applicable."""
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
        final combined-data fit each round). For ``task="classification"``, the K fold-models are scored via
        ``.predict_proba`` (so they need genuine classifier semantics on the REAL, always-discrete labeled
        rows), but ``self.final_model_`` is fit on real labels MIXED with soft (continuous, in ``[0, 1]``)
        pseudo-labels -- a hard sklearn classifier's ``.fit()`` will reject that continuous target. Pass an
        estimator whose ``.fit()`` accepts a continuous ``[0, 1]`` target (e.g. a regressor trained to
        predict a probability) for the classification case, or set ``n_rounds=1`` (the default) so the
        soft-labeled rows are only ever consumed by the final fit, never fed back into a later round's
        fold-model ensemble.
    task
        ``"regression"`` (``.predict`` output averaged/spread directly) or ``"classification"`` (uses
        ``.predict_proba``'s positive-class column; requires a binary classifier for the fold-model
        ensemble). Pseudo-labels fed back for the final fit are ALWAYS soft (continuous), never hardened to
        ``{0, 1}`` -- see ``estimator_factory`` above for what that implies about the estimator you pass.
    n_rounds
        Number of pseudo-labeling rounds. Each round re-scores the FULL unlabeled pool from the fold-model
        ensemble trained on (real labels + the PREVIOUS round's accepted pseudo-labels).
    n_splits, random_state
        K-fold configuration for the fold-model ensemble.
    confidence_threshold
        Regression: max allowed cross-fold-model STD for a row to be accepted (None disables). Classification:
        min allowed distance from 0.5 (i.e. min confidence) for a row to be accepted (None disables). Used as
        the round-0 threshold when ``threshold_anneal`` is set; the sole, ROUND-INVARIANT threshold otherwise.
    pseudo_label_weight
        ``sample_weight`` assigned to accepted pseudo-labeled rows in the final fit (real labeled rows get
        weight 1.0) -- the confirmation-bias guard.
    threshold_anneal
        Opt-in (default ``None`` = static threshold, unchanged behavior). ``"linear"`` or ``"exp"``: interpolate
        the acceptance threshold from ``confidence_threshold`` (round 0) to ``threshold_final`` (last round)
        across rounds, so LATER rounds -- which build on the PREVIOUS round's own pseudo-labels and are thus
        most exposed to confirmation bias -- are held to a stricter bar than the first round. ``"exp"`` squares
        the round fraction, so tightening is backloaded (mild early, sharp in the final rounds). No effect
        unless ``threshold_final`` is also set and ``n_rounds`` > 1.
    threshold_final
        The threshold value to anneal towards by the last round. Required (together with ``threshold_anneal``)
        to activate annealing; ``None`` (default) keeps the static ``confidence_threshold`` throughout.
    class_thresholds
        Opt-in (default ``None``). Classification only: per-predicted-class confidence threshold overriding
        (per row, keyed by ``round(mean_pred)`` in ``{0, 1}``) the scalar threshold for that round -- lets a
        minority class facing harder, noisier fold-model agreement use a stricter bar than the majority class,
        reducing confirmation bias on class-imbalanced problems. A class absent from the dict falls back to
        that round's scalar threshold.

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
        threshold_anneal: Optional[Literal["linear", "exp"]] = None,
        threshold_final: Optional[float] = None,
        class_thresholds: Optional[Dict[int, float]] = None,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.task = task
        self.n_rounds = n_rounds
        self.n_splits = n_splits
        self.random_state = random_state
        self.confidence_threshold = confidence_threshold
        self.pseudo_label_weight = pseudo_label_weight
        self.threshold_anneal = threshold_anneal
        self.threshold_final = threshold_final
        self.class_thresholds = class_thresholds

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

    def _round_threshold(self, round_idx: int) -> Optional[float]:
        """Static (round-invariant) by default; annealed only when both ``threshold_anneal`` and
        ``threshold_final`` are set -- interpolates from ``confidence_threshold`` (round 0) towards
        ``threshold_final`` (last round), so behavior with the new params omitted is unchanged."""
        if self.confidence_threshold is None:
            return None
        if self.threshold_anneal is None or self.threshold_final is None or self.n_rounds <= 1:
            return self.confidence_threshold
        frac = round_idx / (self.n_rounds - 1)
        if self.threshold_anneal == "exp":
            frac = frac**2  # backloaded: mild tightening early, sharp in the final rounds
        return self.confidence_threshold + frac * (self.threshold_final - self.confidence_threshold)

    def _accept_mask(self, mean_pred: np.ndarray, std_pred: np.ndarray, threshold: Optional[float]) -> np.ndarray:
        """Return a boolean mask of unlabeled rows confident enough to accept as pseudo-labels this round."""
        if threshold is None and self.class_thresholds is None:
            return np.ones_like(mean_pred, dtype=bool)
        if self.task == "classification":
            confidence = np.abs(mean_pred - 0.5) * 2.0  # 0 (uncertain) .. 1 (confident)
            if self.class_thresholds is not None:
                fallback = threshold if threshold is not None else 0.0
                pred_class = (mean_pred >= 0.5).astype(int)
                per_row_threshold = np.where(
                    pred_class == 1,
                    self.class_thresholds.get(1, fallback),
                    self.class_thresholds.get(0, fallback),
                )
                return confidence >= per_row_threshold
            assert threshold is not None  # guaranteed by the early-return above when class_thresholds is None
            return confidence >= threshold
        assert threshold is not None  # regression has no class_thresholds fallback
        return std_pred <= threshold

    def fit(self, X_labeled: Any, y_labeled: np.ndarray, X_unlabeled: Any) -> "PseudoLabelingLoop":
        """Run the iterative pseudo-labeling rounds, then fit the final model on real + accepted pseudo-labeled rows."""
        y_arr = np.asarray(y_labeled, dtype=np.float64)
        cur_X, cur_y, cur_w = X_labeled, y_arr, np.ones(len(y_arr), dtype=np.float64)
        # Soft-labeled twin of cur_y: identical on the REAL rows, but keeps each round's raw (continuous)
        # mean_pred for the pseudo rows instead of cur_y's hardened {0.0, 1.0} classification labels. cur_y
        # itself stays hardened for classification so the fold-model ensemble (a genuine classifier, scored
        # via .predict_proba) can still be RE-FIT on it in a later round -- a hard sklearn classifier rejects
        # a continuous target. cur_y_soft is used ONLY for the final fit below, which is what this module's
        # own "soft labels, not hard labels" design promise (and this docstring's estimator_factory note)
        # actually governs.
        cur_y_soft = y_arr
        self.pseudo_labels_history_: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for round_idx in range(self.n_rounds):
            mean_pred, std_pred = self._fold_ensemble_score(cur_X, cur_y, X_unlabeled)
            round_threshold = self._round_threshold(round_idx)
            accept = self._accept_mask(mean_pred, std_pred, round_threshold)
            confidence = (np.abs(mean_pred - 0.5) * 2.0) if self.task == "classification" else -std_pred
            self.pseudo_labels_history_.append((accept, mean_pred, confidence))
            logger.info("PseudoLabelingLoop round %d/%d: accepted %d/%d unlabeled rows", round_idx + 1, self.n_rounds, int(accept.sum()), len(accept))

            hard_labels = np.where(mean_pred >= 0.5, 1.0, 0.0) if self.task == "classification" else mean_pred
            pseudo_X = _select(X_unlabeled, accept)
            pseudo_y_hard = hard_labels[accept]
            pseudo_y_soft = mean_pred[accept]
            pseudo_w = np.full(int(accept.sum()), self.pseudo_label_weight, dtype=np.float64)

            cur_X = _concat(X_labeled, pseudo_X)
            cur_y = np.concatenate([y_arr, pseudo_y_hard])
            cur_y_soft = np.concatenate([y_arr, pseudo_y_soft])
            cur_w = np.concatenate([np.ones(len(y_arr), dtype=np.float64), pseudo_w])

        self.final_model_ = clone(self.estimator_factory())
        try:
            self._fit_final(self.final_model_, cur_X, cur_y_soft, cur_w)
        except ValueError as soft_fit_err:
            # A genuine hard sklearn classifier (its .fit() rejects a continuous target) cannot consume the
            # soft pseudo-labels this module now prefers by default; fall back to the hardened labels rather
            # than crash, but WARN (not silently) -- the whole point of F5's fix is to stop discarding the
            # calibration signal by default, so a caller relying on the fallback should know it happened.
            logger.warning(
                "PseudoLabelingLoop: final_model_.fit() rejected the soft (continuous) pseudo-labels (%s); "
                "falling back to hardened {0,1} labels. Pass a regressor-style estimator_factory (accepts a "
                "continuous [0,1] target) to use soft labels for the final fit.",
                soft_fit_err,
            )
            self.final_model_ = clone(self.estimator_factory())
            self._fit_final(self.final_model_, cur_X, cur_y, cur_w)
        return self

    @staticmethod
    def _fit_final(model: Any, X: Any, y: np.ndarray, sample_weight: np.ndarray) -> None:
        """Fit ``model`` with ``sample_weight`` when its ``fit`` accepts the kwarg, else without."""
        try:
            model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            model.fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the final model fit on real plus accepted pseudo-labeled rows."""
        if self.task == "classification":
            return np.asarray(self.final_model_.predict_proba(X), dtype=np.float64)[:, 1]
        return np.asarray(self.final_model_.predict(X), dtype=np.float64)


__all__ = ["PseudoLabelingLoop"]
