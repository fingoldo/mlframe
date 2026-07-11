"""``greedy_backward_elimination``: iteratively drop the single feature whose removal most improves CV score.

Source: dd_1st_pover-t-tests.md -- permutation importance used not just to rank but to actually decide
removal: "removed the ones for which we registered a score improvement" when shuffled/dropped. Distinct from
mlframe's existing `RFECV` (drops the worst-RANKED feature per round by importance) and
`unanimous_permutation_prune` (drops any feature permutation fails to improve in EVERY fold): this evaluates
removing EACH remaining feature via fresh CV, removes whichever single removal most improves the mean CV
score, and repeats until no remaining removal helps -- a directly score-driven search rather than an
importance-proxy or a fixed unanimity rule.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def _cv_score(estimator, X: pd.DataFrame, y: np.ndarray, cv, scoring: Callable[[np.ndarray, np.ndarray], float]) -> float:
    scores = []
    for train_idx, test_idx in cv.split(X):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(scoring(y[test_idx], preds))
    return float(np.mean(scores))


def greedy_backward_elimination(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    cv: Optional[object] = None,
    min_features: int = 1,
    tol: float = 0.0,
) -> List[str]:
    """Repeatedly remove the single feature whose removal most improves mean CV ``scoring``, HIGHER is better.

    Parameters
    ----------
    estimator
        Unfitted sklearn-compatible estimator (cloned per fold/candidate).
    X
        ``(n, d)`` feature frame.
    y
        ``(n,)`` target.
    scoring
        ``scoring(y_true, y_pred) -> float``, HIGHER is better.
    cv
        sklearn-style CV splitter; defaults to ``KFold(n_splits=5, shuffle=True, random_state=0)``.
    min_features
        Stop once this many features remain, even if a further removal would still help.
    tol
        A removal is accepted only if it improves the mean CV score by more than ``tol`` (default ``0.0``:
        any improvement counts).

    Returns
    -------
    list of str
        Surviving column names, in original order.
    """
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=0)

    remaining = list(X.columns)
    current_score = _cv_score(estimator, X[remaining], y, cv, scoring)

    while len(remaining) > min_features:
        best_candidate = None
        best_score = current_score
        for col in remaining:
            candidate_cols = [c for c in remaining if c != col]
            score = _cv_score(estimator, X[candidate_cols], y, cv, scoring)
            if score > best_score + tol:
                best_score = score
                best_candidate = col

        if best_candidate is None:
            break

        remaining.remove(best_candidate)
        current_score = best_score

    return remaining


__all__ = ["greedy_backward_elimination"]
