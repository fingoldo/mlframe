"""``forward_select``: plain greedy forward feature selection by CV score.

Distinct from MRMR's greedy loop (MI/redundancy-driven, not CV-score-driven) and from RFECV's SFFS swap pass
(a refinement pass over an already-narrowed subset, not a from-scratch greedy build). Starts from an empty
subset and repeatedly adds the single candidate feature giving the largest cross-validated score
improvement, stopping when no remaining candidate improves the score by at least ``min_improvement`` or
``max_features`` is reached.

Cost: O(d^2) model fits in the worst case (every remaining candidate is tried at every step) -- fine for the
tens-to-low-hundreds of already-screened candidates this is meant to run on (e.g. after a Boruta-style
shadow-feature filter has cut a few thousand raw features down to a few dozen), not intended as a
first-pass filter over the full raw feature set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class MarginalGainStep:
    """One round of the greedy loop, for early-stop/exhaustion diagnostics."""

    candidate: Any
    improvement: float
    p_value: Optional[float]
    significant: bool


@dataclass
class ForwardSelectReport:
    """Per-step diagnostics for a ``forward_select`` run (only built when ``return_report=True``)."""

    steps: List[MarginalGainStep] = field(default_factory=list)
    stopped_early: bool = False


def forward_select(
    X: Any,
    y: np.ndarray,
    estimator_factory: Callable[[], Any],
    scoring: Optional[str] = None,
    cv: int = 5,
    max_features: Optional[int] = None,
    min_improvement: float = 0.0,
    candidate_features: Optional[Sequence[Any]] = None,
    initial_selected: Optional[Sequence[Any]] = None,
    patience: Optional[int] = None,
    significance_level: float = 0.05,
    return_report: bool = False,
) -> Union[List[Any], Tuple[List[Any], ForwardSelectReport]]:
    """Greedily grow a feature subset one column at a time by best CV-score marginal improvement.

    Parameters
    ----------
    X
        Feature frame (pandas DataFrame, columns addressable by name) or ndarray (columns addressed by
        integer index).
    y
        Target array.
    estimator_factory
        Zero-arg callable returning a fresh unfitted estimator, called fresh for every (subset, cv-fold)
        evaluation.
    scoring
        sklearn scorer name (e.g. ``"neg_mean_squared_error"``, ``"roc_auc"``); None uses the estimator's
        own default scorer.
    cv
        Number of CV folds per candidate evaluation.
    max_features
        Stop once the selected subset reaches this size (default: no cap, all candidates).
    min_improvement
        Stop once the best remaining candidate's score improvement over the current subset falls below this
        threshold (default 0.0: stop on any non-improvement).
    candidate_features
        Restrict the search to this column subset (default: all of ``X``'s columns).
    initial_selected
        Column(s) always included in every trial subset and never candidates for removal or re-selection --
        e.g. a stacking meta-model's fixed core of first-level OOF predictions, with only raw features
        greedily forward-selected on top (the "raw-feature-augmented meta-model" pattern). ``None``
        (default) preserves the original empty-start behavior exactly.
    patience
        Opt-in early-stop: once the best remaining candidate's per-fold CV-score improvement over the
        current subset is statistically indistinguishable from noise (paired one-sided t-test, candidate
        folds vs. current-subset folds, Bonferroni-corrected p-value > ``significance_level`` -- corrected
        by dividing by the number of remaining candidates, since each round already picks the best of many
        comparisons) for this many consecutive rounds, stop the loop instead of continuing to exhaustion.
        ``None`` (default) disables this check entirely -- the loop always runs to
        ``max_features``/candidate exhaustion exactly as before, byte-for-byte.
    significance_level
        Pre-correction p-value threshold for the ``patience`` noise test (default 0.05). Unused when
        ``patience`` is None.
    return_report
        When True, return ``(selected, report)`` instead of just ``selected``. ``report`` is a
        ``ForwardSelectReport`` with one ``MarginalGainStep`` per round (the round's best candidate, its
        mean CV-score improvement, and -- when computable -- the paired-t-test p-value/significance flag
        used by ``patience``). Default False preserves the original ``List[Any]`` return type exactly.

    Returns
    -------
    list, or (list, ForwardSelectReport) when ``return_report=True``
        Selected column names/indices, in the order they were added (``initial_selected`` columns first, in
        the order given, followed by greedily-added candidates).
    """
    from sklearn.model_selection import cross_val_score

    is_frame = hasattr(X, "columns")
    all_candidates = list(candidate_features) if candidate_features is not None else (list(X.columns) if is_frame else list(range(np.asarray(X).shape[1])))

    def _subset(cols: List[Any]) -> Any:
        """Column-select ``cols`` from X, working for both a DataFrame and a bare ndarray."""
        return X[cols] if is_frame else np.asarray(X)[:, cols]

    diagnostics_needed = patience is not None or return_report

    selected: List[Any] = list(initial_selected) if initial_selected is not None else []
    remaining = [c for c in all_candidates if c not in selected]
    current_fold_scores: Optional[np.ndarray] = None
    if selected:
        baseline_model = estimator_factory()
        baseline_fold_scores = cross_val_score(baseline_model, _subset(selected), y, cv=cv, scoring=scoring)
        best_score = float(np.mean(baseline_fold_scores))
        if diagnostics_needed:
            current_fold_scores = baseline_fold_scores
    else:
        best_score = -np.inf
    cap = (max_features if max_features is not None else len(all_candidates)) + len(selected)

    report = ForwardSelectReport()
    non_significant_streak = 0

    while remaining and len(selected) < cap:
        trial_scores: dict[Any, float] = {}
        trial_fold_scores: dict[Any, np.ndarray] = {}
        for candidate in remaining:
            trial_cols = [*selected, candidate]
            model = estimator_factory()
            fold_scores = cross_val_score(model, _subset(trial_cols), y, cv=cv, scoring=scoring)
            trial_scores[candidate] = float(np.mean(fold_scores))
            if diagnostics_needed:
                trial_fold_scores[candidate] = fold_scores

        best_candidate = max(trial_scores, key=lambda c: trial_scores[c])
        improvement = trial_scores[best_candidate] - best_score
        if selected and improvement < min_improvement:
            break

        p_value: Optional[float] = None
        significant = True
        if diagnostics_needed and current_fold_scores is not None:
            from scipy.stats import ttest_rel

            best_fold_scores = trial_fold_scores[best_candidate]
            if np.allclose(best_fold_scores, current_fold_scores):
                p_value, significant = 1.0, False
            else:
                _, p_value = ttest_rel(best_fold_scores, current_fold_scores, alternative="greater")
                # Bonferroni-correct for picking the best of `len(remaining)` candidates each round --
                # without this, "best-of-many noise columns" is significant far more often than
                # `significance_level` alone implies (the classic post-selection multiple-comparisons bias).
                significant = bool(p_value <= significance_level / len(remaining))

        if diagnostics_needed:
            report.steps.append(MarginalGainStep(candidate=best_candidate, improvement=improvement, p_value=p_value, significant=significant))

        if patience is not None:
            non_significant_streak = 0 if significant else non_significant_streak + 1
            if non_significant_streak >= patience:
                report.stopped_early = True
                break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = trial_scores[best_candidate]
        if diagnostics_needed:
            current_fold_scores = trial_fold_scores[best_candidate]

    if return_report:
        return selected, report
    return selected


__all__ = ["forward_select", "ForwardSelectReport", "MarginalGainStep"]
