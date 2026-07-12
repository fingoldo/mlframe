"""COMPETITION/EXPLORATORY-ONLY utility. NOT for production use.

Implements the "threshold-based manual post-hoc probability correction" trick
documented in ``MLFRAME_IDEAS_competitions.md`` (source: 4th place,
home-credit-default-risk — "if you correct your prediction for revolving loan
that is over 0.4 by 0.8, it will boost your auc").

``ThresholdRangeRescaler`` formalizes that ad hoc manual guess into an
auto-searched grid over ``(subgroup, threshold, multiplier)`` combinations,
each scored by K-fold cross-validated AUC (or any user-supplied metric) on the
corrected predictions, keeping only the combination(s) that improve CV score
over doing nothing.

**This is exactly the "magic number" CV-overfitting hack the tracker entry
itself warns about.** Grid-searching a multiplicative correction directly
against a CV metric on the very predictions being corrected can trivially
latch onto CV noise rather than a genuine systematic miscalibration,
especially with many subgroup/threshold/multiplier combinations and few CV
folds. It is appropriate only for Kaggle-style leaderboard chasing where the
CV split approximates the leaderboard split; it has no place in a production
calibration pipeline. Never import this module from production mlframe code
paths and never wire it into any default pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class ThresholdCorrection:
    """A single fitted ``(subgroup, threshold, multiplier)`` correction rule."""

    subgroup: str
    threshold: float
    multiplier: float
    cv_score: float


@dataclass
class ThresholdRangeRescalerResult:
    """Diagnostics returned alongside the fitted corrections."""

    baseline_cv_score: float
    final_cv_score: float
    corrections: list[ThresholdCorrection] = field(default_factory=list)


class ThresholdRangeRescaler:
    """COMPETITION-ONLY. Not for production use.

    Fits a small, greedily-built set of piecewise multiplicative corrections:
    for predictions belonging to a given covariate-defined ``subgroup`` (a
    boolean mask) and exceeding a given ``threshold``, multiply the prediction
    by a ``multiplier``. Both the subgroup, the threshold and the multiplier
    are chosen by grid search maximizing a cross-validated metric (AUC by
    default).

    See the module docstring for why this is a CV-overfitting-prone hack
    rather than a principled calibration method — use only for competition
    leaderboard chasing, never in production.

    Parameters
    ----------
    thresholds:
        Grid of prediction thresholds to try (a correction applies to
        ``preds > threshold`` within the subgroup).
    multipliers:
        Grid of multiplicative correction factors to try. Include ``1.0`` so
        "no correction" is always a candidate — this is what keeps the honest
        no-signal case a no-op instead of always forcing some multiplier.
    n_splits:
        Number of CV folds used to score each candidate correction.
    max_corrections:
        Maximum number of corrections to greedily stack (each round searches
        the grid again on the residual of previously applied corrections).
        The source idea describes "a small set" of corrections, not one.
    min_improvement:
        Minimum absolute CV-score improvement required to accept a candidate
        correction; stops the greedy search once no candidate clears the bar.
    metric_fn:
        Callable ``(y_true, y_score) -> float`` to maximize. Defaults to
        ``roc_auc_score``.
    random_state:
        Seed for the ``StratifiedKFold`` shuffle.

    Attributes (post-fit)
    ----------------------
    corrections_ : list[ThresholdCorrection]
        The accepted corrections, in application order.
    baseline_cv_score_ : float
        CV score of the uncorrected predictions.
    final_cv_score_ : float
        CV score after applying all accepted corrections.
    """

    def __init__(
        self,
        thresholds: np.ndarray,
        multipliers: np.ndarray,
        n_splits: int = 5,
        max_corrections: int = 3,
        min_improvement: float = 1e-4,
        metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
        random_state: int | None = 0,
    ) -> None:
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.multipliers = np.asarray(multipliers, dtype=np.float64)
        self.n_splits = n_splits
        self.max_corrections = max_corrections
        self.min_improvement = min_improvement
        self.metric_fn = metric_fn
        self.random_state = random_state

        self.corrections_: list[ThresholdCorrection] = []
        self.baseline_cv_score_: float = float("nan")
        self.final_cv_score_: float = float("nan")

    def _cv_score(self, preds: np.ndarray, y: np.ndarray) -> float:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_scores = []
        for _, test_idx in skf.split(preds, y):
            y_test = y[test_idx]
            if len(np.unique(y_test)) < 2:
                continue
            fold_scores.append(self.metric_fn(y_test, preds[test_idx]))
        if not fold_scores:
            raise ValueError("could not compute CV score: no fold had both classes present")
        return float(np.mean(fold_scores))

    def fit(
        self,
        preds: np.ndarray,
        y: np.ndarray,
        subgroups: dict[str, np.ndarray],
    ) -> "ThresholdRangeRescaler":
        """Grid-search and greedily stack the best-scoring corrections.

        Parameters
        ----------
        preds:
            1-D array of predicted probabilities in ``[0, 1]``.
        y:
            1-D array of binary ground-truth labels, same length as ``preds``.
        subgroups:
            Mapping of subgroup name -> boolean mask (same length as
            ``preds``) defining covariate-based subpopulations to search
            corrections over (e.g. ``{"revolving_loan": df["loan_type"] ==
            "revolving"}``).
        """
        preds_arr = np.asarray(preds, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        if preds_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError("preds and y must be 1-D")
        if preds_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("preds and y must have the same length")
        if preds_arr.shape[0] == 0:
            raise ValueError("preds must be non-empty")
        for name, mask in subgroups.items():
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape[0] != preds_arr.shape[0]:
                raise ValueError(f"subgroup mask {name!r} length does not match preds")

        working = preds_arr.copy()
        self.baseline_cv_score_ = self._cv_score(working, y_arr)
        current_score = self.baseline_cv_score_
        self.corrections_ = []

        for _ in range(self.max_corrections):
            best_gain = self.min_improvement
            best_choice: ThresholdCorrection | None = None
            best_corrected: np.ndarray | None = None

            for name, mask in subgroups.items():
                mask_arr = np.asarray(mask, dtype=bool)
                if not mask_arr.any():
                    continue
                for threshold in self.thresholds:
                    hit = mask_arr & (working > threshold)
                    if not hit.any():
                        continue
                    for multiplier in self.multipliers:
                        if multiplier == 1.0:
                            continue
                        candidate = working.copy()
                        candidate[hit] = np.clip(candidate[hit] * multiplier, 0.0, 1.0)
                        score = self._cv_score(candidate, y_arr)
                        gain = score - current_score
                        if gain > best_gain:
                            best_gain = gain
                            best_choice = ThresholdCorrection(
                                subgroup=name, threshold=float(threshold), multiplier=float(multiplier), cv_score=score
                            )
                            best_corrected = candidate

            if best_choice is None or best_corrected is None:
                break

            self.corrections_.append(best_choice)
            working = best_corrected
            current_score = best_choice.cv_score

        self.final_cv_score_ = current_score
        return self

    def transform(self, preds: np.ndarray, subgroups: dict[str, np.ndarray]) -> np.ndarray:
        """Apply the fitted corrections, in the order they were accepted."""
        corrected = np.asarray(preds, dtype=np.float64).copy()
        for correction in self.corrections_:
            mask = np.asarray(subgroups[correction.subgroup], dtype=bool)
            hit = mask & (corrected > correction.threshold)
            corrected[hit] = np.clip(corrected[hit] * correction.multiplier, 0.0, 1.0)
        return np.asarray(corrected, dtype=np.float64)

    def fit_transform(
        self,
        preds: np.ndarray,
        y: np.ndarray,
        subgroups: dict[str, np.ndarray],
    ) -> np.ndarray:
        self.fit(preds, y, subgroups)
        return self.transform(preds, subgroups)

    def result(self) -> ThresholdRangeRescalerResult:
        """Return a snapshot of fit diagnostics (baseline/final CV score, corrections)."""
        return ThresholdRangeRescalerResult(
            baseline_cv_score=self.baseline_cv_score_,
            final_cv_score=self.final_cv_score_,
            corrections=list(self.corrections_),
        )
