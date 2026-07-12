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


def _cv_score_repeated(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    n_splits: int,
    n_repeats: int,
    seed_base: int,
) -> float:
    """Average ``_cv_score`` across ``n_repeats`` independently-shuffled ``KFold`` splits.

    Each repeat gets its own ``random_state`` (``seed_base + repeat_idx``) so the removal decision reflects
    the score across several splits rather than a single noisy one.
    """
    repeat_scores = []
    for repeat_idx in range(n_repeats):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed_base + repeat_idx)
        repeat_scores.append(_cv_score(estimator, X, y, cv, scoring))
    return float(np.mean(repeat_scores))


def greedy_backward_elimination(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    cv: Optional[object] = None,
    min_features: int = 1,
    tol: float = 0.0,
    n_repeats: int = 1,
    seed_base: int = 0,
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
        sklearn-style CV splitter; defaults to ``KFold(n_splits=5, shuffle=True, random_state=0)``. Ignored
        when ``n_repeats > 1`` (seed-averaging drives its own splitters instead, see ``n_repeats``).
    min_features
        Stop once this many features remain, even if a further removal would still help.
    tol
        A removal is accepted only if it improves the mean CV score by more than ``tol`` (default ``0.0``:
        any improvement counts).
    n_repeats
        Opt-in seed-averaging: when ``> 1``, every removal decision (both the current baseline and each
        candidate) averages ``_cv_score`` over ``n_repeats`` independently-shuffled ``KFold`` splits
        (``n_splits`` taken from ``cv.get_n_splits()`` if ``cv`` is given, else 5) instead of a single CV
        run, so a single noisy split can't wrongly keep a weak-but-real feature or drop a lucky noise
        feature. Default ``1`` reproduces the original single-run behavior bit-for-bit (``cv`` is used as
        given, and ``seed_base`` has no effect).
    seed_base
        First ``random_state`` used when ``n_repeats > 1``; repeat ``i`` uses ``seed_base + i``. Unused when
        ``n_repeats == 1``.

    Returns
    -------
    list of str
        Surviving column names, in original order.
    """
    if n_repeats > 1:
        n_splits = cv.get_n_splits() if cv is not None and hasattr(cv, "get_n_splits") else 5

        def score_fn(frame: pd.DataFrame) -> float:
            return _cv_score_repeated(estimator, frame, y, scoring, n_splits=n_splits, n_repeats=n_repeats, seed_base=seed_base)

    else:
        if cv is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=0)

        def score_fn(frame: pd.DataFrame) -> float:
            return _cv_score(estimator, frame, y, cv, scoring)

    remaining = list(X.columns)
    current_score = score_fn(X[remaining])

    while len(remaining) > min_features:
        best_candidate = None
        best_score = current_score
        for col in remaining:
            candidate_cols = [c for c in remaining if c != col]
            score = score_fn(X[candidate_cols])
            if score > best_score + tol:
                best_score = score
                best_candidate = col

        if best_candidate is None:
            break

        remaining.remove(best_candidate)
        current_score = best_score

    return remaining


__all__ = ["greedy_backward_elimination"]
