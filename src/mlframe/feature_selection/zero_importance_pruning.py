"""``iterative_zero_importance_pruning``: cheap batch-drop-zero-importance pre-filter.

Source: dd_2nd_pover-t-tests.md -- "The `find_exclude` method iteratively retrains a model, drops zero-
importance features, and re-evaluates 5-fold CV log loss, repeating until no further features can be dropped,
then keeps the feature-exclusion set with lowest CV log loss." Distinct from mlframe's existing selectors:
RFECV drops the single worst-ranked feature per round; `greedy_backward_elimination` evaluates removing EACH
candidate via a fresh CV pass (O(features^2 x folds)). This drops the WHOLE batch of zero/near-zero native
`feature_importances_` features per round in one refit, tracking CV score only once per round and stopping on
degradation -- an O(features x rounds) fast pre-filter for tree ensembles with cheap native importances,
meant to run BEFORE the heavier MRMR/RFECV passes on a huge feature set, not to replace them.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def _cv_score(estimator, X: pd.DataFrame, y: np.ndarray, cv, scoring: Callable[[np.ndarray, np.ndarray], float]) -> float:
    """Fit a cloned estimator on each CV fold and return the mean out-of-fold score."""
    row_select = (lambda idx: X.iloc[idx]) if hasattr(X, "iloc") else (lambda idx: X[idx])
    scores = []
    for train_idx, test_idx in cv.split(X):
        model = clone(estimator)
        model.fit(row_select(train_idx), y[train_idx])
        preds = model.predict(row_select(test_idx))
        scores.append(scoring(y[test_idx], preds))
    return float(np.mean(scores))


def iterative_zero_importance_pruning(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    cv: Optional[object] = None,
    importance_threshold: float = 0.0,
    max_rounds: int = 20,
    importance_fn: Optional[Callable[[Any, pd.DataFrame, np.ndarray], np.ndarray]] = None,
) -> List[str]:
    """Repeatedly drop the WHOLE batch of near-zero-importance features per round, stopping on CV degradation.

    Parameters
    ----------
    estimator
        Unfitted sklearn-compatible estimator. Cloned per fold/round. Must expose ``feature_importances_``
        after ``fit`` (e.g. any tree ensemble) unless ``importance_fn`` is supplied.
    X
        ``(n, d)`` feature frame.
    y
        ``(n,)`` target.
    scoring
        ``scoring(y_true, y_pred) -> float``, HIGHER is better.
    cv
        sklearn-style CV splitter; defaults to ``KFold(n_splits=5, shuffle=True, random_state=0)``.
    importance_threshold
        Features with importance ``<= importance_threshold`` are candidates for the batch drop.
        Default ``0.0`` (exact zero-importance only, matching the source's own criterion).
    max_rounds
        Safety cap on the number of drop-and-refit rounds.
    importance_fn
        Optional ``importance_fn(fitted_estimator, X_round, y) -> np.ndarray`` returning one importance value
        per column of ``X_round`` (same order). When omitted (default), the native ``feature_importances_``
        of the fitted estimator is used, preserving prior behavior exactly. Supply this for estimators
        without a reliable/meaningful native importance -- e.g. linear models with no ``feature_importances_``
        at all, or tree ensembles whose native importance is biased toward high-cardinality columns -- typically
        a permutation-importance or SHAP-based callable (e.g. wrapping ``sklearn.inspection.permutation_importance``).

    Returns
    -------
    list of str
        The best-scoring surviving feature set seen across all rounds (not necessarily the LAST round's set --
        the round with the highest CV score is kept, matching the source's "keeps the exclusion set with
        lowest CV log loss" convention, generalized to "highest ``scoring``").
    """
    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=0)

    remaining = list(X.columns)
    best_score = _cv_score(estimator, X[remaining], y, cv, scoring)
    best_remaining = list(remaining)

    for _ in range(max_rounds):
        full_fit_estimator = clone(estimator).fit(X[remaining], y)
        if importance_fn is None:
            importances = np.asarray(full_fit_estimator.feature_importances_)
        else:
            importances = np.asarray(importance_fn(full_fit_estimator, X[remaining], y))
        zero_mask = importances <= importance_threshold
        if not zero_mask.any():
            break

        candidate_remaining = [col for col, is_zero in zip(remaining, zero_mask) if not is_zero]
        if not candidate_remaining:
            break  # never drop every feature -- degenerate case, stop here.

        candidate_score = _cv_score(estimator, X[candidate_remaining], y, cv, scoring)
        remaining = candidate_remaining
        if candidate_score > best_score:
            best_score = candidate_score
            best_remaining = list(remaining)

    return best_remaining


__all__ = ["iterative_zero_importance_pruning"]
