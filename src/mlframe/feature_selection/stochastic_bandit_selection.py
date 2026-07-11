"""``stochastic_bandit_selection``: adaptive-weighted random-subset feature selection.

Source: 2nd_mercedes-benz-greener-manufacturing.md -- iterative process: randomly sample small feature
subsets weighted by an adaptive probability vector; if CV score beats a moving average, increase weight of
those features (and vice versa); promote features whose weight crosses a threshold into a permanent
"top_feats" pool; repeat for many epochs; final feature set = epoch achieving best CV. A multi-armed-bandit-
style alternative to MRMR/RFECV's deterministic greedy search: cheap per-epoch (a small random subset + one
quick low-complexity model fit), useful when the feature-target relationship is noisy enough that greedy
elimination can get stuck in a locally-good-but-globally-suboptimal subset.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

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


def _stochastic_bandit_selection_core(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    subset_size: int,
    n_epochs: int,
    cv: Optional[object],
    up_factor: float,
    down_factor: float,
    lock_in_threshold: float,
    moving_average_window: int,
    random_state: int,
) -> Tuple[List[str], List[str]]:
    """Shared search loop. Returns ``(best_subset, locked_in_feats)`` -- the single-seed public API only
    exposes ``best_subset``; the multi-seed ensemble mode (``stochastic_bandit_selection_ensemble``) also
    needs each run's permanently locked-in "top_feats" pool to compute cross-seed stability.
    """
    if cv is None:
        cv = KFold(n_splits=3, shuffle=True, random_state=0)
    rng = np.random.default_rng(random_state)

    columns = list(X.columns)
    n_features = len(columns)
    weights = np.ones(n_features, dtype=np.float64)
    locked_in = np.zeros(n_features, dtype=bool)

    recent_scores: List[float] = []
    best_score = -np.inf
    best_subset: List[str] = columns[:subset_size]

    for _ in range(n_epochs):
        locked_idx = np.flatnonzero(locked_in)
        n_random_needed = max(0, subset_size - len(locked_idx))
        candidate_idx = np.flatnonzero(~locked_in)
        if n_random_needed > 0 and len(candidate_idx) > 0:
            probs = weights[candidate_idx] / weights[candidate_idx].sum()
            n_pick = min(n_random_needed, len(candidate_idx))
            picked = rng.choice(candidate_idx, size=n_pick, replace=False, p=probs)
            subset_idx = np.concatenate([locked_idx, picked])
        else:
            subset_idx = locked_idx[:subset_size] if len(locked_idx) > subset_size else locked_idx

        subset_cols = [columns[i] for i in subset_idx]
        score = _cv_score(estimator, X[subset_cols], y, cv, scoring)

        baseline = float(np.mean(recent_scores[-moving_average_window:])) if recent_scores else score
        if score >= baseline:
            weights[subset_idx] *= up_factor
        else:
            weights[subset_idx] *= down_factor
        weights = np.clip(weights, 1e-6, None)
        locked_in |= weights >= lock_in_threshold

        recent_scores.append(score)
        if score > best_score:
            best_score = score
            best_subset = subset_cols

    locked_in_feats = [columns[i] for i in np.flatnonzero(locked_in)]
    return best_subset, locked_in_feats


def stochastic_bandit_selection(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    subset_size: int,
    n_epochs: int = 200,
    cv: Optional[object] = None,
    up_factor: float = 1.05,
    down_factor: float = 0.97,
    lock_in_threshold: float = 3.0,
    moving_average_window: int = 10,
    random_state: int = 0,
) -> List[str]:
    """Adaptive-weighted random-subset search: return the feature subset with the best CV score seen.

    Parameters
    ----------
    estimator
        Unfitted sklearn-compatible estimator (a cheap/low-complexity one is recommended, per the source's
        own convention -- this is a per-epoch quick-fit search, not a single expensive final fit).
    X, y
        Feature frame and target.
    scoring
        ``scoring(y_true, y_pred) -> float``, HIGHER is better.
    subset_size
        Number of features sampled per epoch.
    n_epochs
        Number of random-subset trials.
    cv
        sklearn-style CV splitter; defaults to ``KFold(n_splits=3, shuffle=True, random_state=0)``.
    up_factor, down_factor
        Multiplicative weight adjustment applied to every feature in a subset that beat/missed the running
        moving-average baseline.
    lock_in_threshold
        A feature whose weight (relative to the initial uniform weight) exceeds this is always included in
        every subsequent subset (the source's "top_feats" pool).
    moving_average_window
        Number of recent epoch scores averaged into the baseline a new epoch's score is compared against.
    random_state
        Seed for the subset sampler.

    Returns
    -------
    list of str
        The feature subset (as column names) achieving the best CV score across all epochs.

    See Also
    --------
    stochastic_bandit_selection_ensemble.stochastic_bandit_selection_ensemble
        Opt-in multi-seed variant that unions several independent runs' locked-in feature pools and reports
        a per-feature cross-seed selection-agreement (stability) diagnostic.
    """
    best_subset, _ = _stochastic_bandit_selection_core(
        estimator=estimator,
        X=X,
        y=y,
        scoring=scoring,
        subset_size=subset_size,
        n_epochs=n_epochs,
        cv=cv,
        up_factor=up_factor,
        down_factor=down_factor,
        lock_in_threshold=lock_in_threshold,
        moving_average_window=moving_average_window,
        random_state=random_state,
    )
    return best_subset


__all__ = ["stochastic_bandit_selection"]
