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

from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def _cv_score(estimator, X: pd.DataFrame, y_arr: np.ndarray, folds, scoring: Callable[[np.ndarray, np.ndarray], float]) -> float:
    """``folds`` is a precomputed ``[(train_idx, test_idx), ...]`` list, not a CV splitter to call ``.split()`` on.

    ``greedy_backward_elimination``'s O(d^2) removal search calls this once per remaining column per round, always
    with the SAME row count (only columns change) -- ``cv.split(X)`` reshuffles and re-derives the identical fold
    index arrays every single call, pure wasted work since the row count (and therefore the split) never changes
    across candidates. The caller now computes ``folds`` ONCE and passes it in; see ``greedy_backward_elimination``.
    """
    row_select = (lambda idx: X.iloc[idx]) if hasattr(X, "iloc") else (lambda idx: X[idx])
    scores = []
    for train_idx, test_idx in folds:
        model = clone(estimator)
        model.fit(row_select(train_idx), y_arr[train_idx])
        preds = model.predict(row_select(test_idx))
        scores.append(scoring(y_arr[test_idx], preds))
    return float(np.mean(scores))


def _cv_score_repeated(
    estimator: Any,
    X: pd.DataFrame,
    y_arr: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    repeat_folds: list,
) -> float:
    """Average ``_cv_score`` across the precomputed per-repeat fold sets in ``repeat_folds``.

    Each repeat set was built from its own ``random_state`` (``seed_base + repeat_idx``) so the removal decision
    reflects the score across several splits rather than a single noisy one; computed once by the caller (see
    ``greedy_backward_elimination``) since the row count -- and therefore every repeat's split -- never changes
    across the O(d^2) column-drop candidates.
    """
    return float(np.mean([_cv_score(estimator, X, y_arr, folds, scoring) for folds in repeat_folds]))


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
    # Coerce to a plain ndarray ONCE so bare ``y_arr[idx]`` is unambiguously positional: a pd.Series ``y`` with a
    # non-default (gapped) index -- e.g. after an upstream row filter that didn't reset_index() -- makes bare
    # bracket indexing resolve train_idx/test_idx (KFold's 0-based POSITIONS) as LABELS instead, raising a
    # spurious KeyError once the index has any gaps relative to a dense 0..n-1 range.
    y_arr = np.asarray(y)
    n = len(X)

    if n_repeats > 1:
        n_splits = cv.get_n_splits() if cv is not None and hasattr(cv, "get_n_splits") else 5
        # Precompute every repeat's fold index arrays ONCE: the row count (hence every split) is invariant across
        # the O(d^2) column-drop candidates below, so re-deriving them per candidate is pure wasted shuffling.
        repeat_folds = [list(KFold(n_splits=n_splits, shuffle=True, random_state=seed_base + repeat_idx).split(np.empty(n))) for repeat_idx in range(n_repeats)]

        def score_fn(frame: pd.DataFrame) -> float:
            return _cv_score_repeated(estimator, frame, y_arr, scoring, repeat_folds)

    else:
        if cv is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=0)
        folds = list(cv.split(np.empty(n)))

        def score_fn(frame: pd.DataFrame) -> float:
            return _cv_score(estimator, frame, y_arr, folds, scoring)

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
