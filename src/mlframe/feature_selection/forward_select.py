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

from typing import Any, Callable, List, Optional, Sequence

import numpy as np


def forward_select(
    X: Any,
    y: np.ndarray,
    estimator_factory: Callable[[], Any],
    scoring: Optional[str] = None,
    cv: int = 5,
    max_features: Optional[int] = None,
    min_improvement: float = 0.0,
    candidate_features: Optional[Sequence[Any]] = None,
) -> List[Any]:
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

    Returns
    -------
    list
        Selected column names/indices, in the order they were added.
    """
    from sklearn.model_selection import cross_val_score

    is_frame = hasattr(X, "columns")
    all_candidates = list(candidate_features) if candidate_features is not None else (list(X.columns) if is_frame else list(range(np.asarray(X).shape[1])))

    def _subset(cols: List[Any]) -> Any:
        return X[cols] if is_frame else np.asarray(X)[:, cols]

    selected: List[Any] = []
    remaining = list(all_candidates)
    best_score = -np.inf
    cap = max_features if max_features is not None else len(all_candidates)

    while remaining and len(selected) < cap:
        trial_scores: dict[Any, float] = {}
        for candidate in remaining:
            trial_cols = selected + [candidate]
            model = estimator_factory()
            scores = cross_val_score(model, _subset(trial_cols), y, cv=cv, scoring=scoring)
            trial_scores[candidate] = float(np.mean(scores))

        best_candidate = max(trial_scores, key=lambda c: trial_scores[c])
        improvement = trial_scores[best_candidate] - best_score
        if selected and improvement < min_improvement:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = trial_scores[best_candidate]

    return selected


__all__ = ["forward_select"]
