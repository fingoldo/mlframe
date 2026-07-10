"""``unanimous_permutation_prune``: drop a feature only if permuting it fails to improve EVERY CV fold.

Source: 2nd_ieee-cis-fraud-detection.md -- kept a feature only if permuting it did NOT improve ANY of the 4
walk-forward CV models' predictions (strict multi-fold agreement), iterated to convergence. This is a more
CONSERVATIVE variant of standard mean-permutation-importance pruning: a feature whose permutation HELPS the
metric in even one fold (a noisy/unstable time-split signal) survives, since dropping it risks discarding a
feature that's genuinely useful in some regimes even if its AVERAGE importance looks weak. Standalone rather
than wired into `RFECV`'s existing `VotesAggregation`/fold-aggregation machinery (which already has 8
established vote rules and is exercised by a large existing test suite) -- this keeps the new, more aggressive
pruning criterion isolated and independently testable rather than risking regressions in that shared,
heavily-used aggregation path.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np


def unanimous_permutation_prune(
    X: Any,
    y: np.ndarray,
    estimator_factory: Callable[[], Any],
    cv_splits: Sequence,
    scoring: Optional[Callable[[Any, Any], float]] = None,
    n_repeats: int = 5,
    max_iterations: int = 10,
    random_state: int = 0,
    feature_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Iteratively drop features that fail to improve the metric in EVERY fold when permuted.

    Parameters
    ----------
    X
        Feature matrix (pandas DataFrame with named columns, or ndarray -- ``feature_names`` required for
        ndarray input).
    y
        Target.
    estimator_factory
        Callable returning a fresh, unfitted sklearn-compatible estimator.
    cv_splits
        Sequence of ``(train_idx, val_idx)`` pairs (walk-forward/time-split CV -- caller-constructed, e.g.
        via ``sklearn.model_selection.TimeSeriesSplit(...).split(X)``).
    scoring
        ``scoring(y_true, y_pred) -> float``, HIGHER is better. Defaults to negative RMSE for continuous
        targets (regression); pass an explicit metric for classification.
    n_repeats
        Number of permutation repeats per feature per fold (averaged for that fold's importance estimate).
    max_iterations
        Cap on prune-and-refit iterations (stops earlier once no feature qualifies for removal).
    random_state
        Seed for the permutation shuffles.

    Returns
    -------
    list of str
        Surviving feature names after iterating to convergence (or ``max_iterations``).
    """
    import pandas as pd
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer, mean_squared_error

    if scoring is None:
        scoring = lambda y_true, y_pred: -float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))  # noqa: E731

    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
    else:
        if feature_names is None:
            raise ValueError("unanimous_permutation_prune: feature_names is required when X is not a DataFrame")
        names = list(feature_names)
        X = pd.DataFrame(np.asarray(X), columns=names)

    sk_scorer = make_scorer(scoring, greater_is_better=True)
    rng = np.random.default_rng(random_state)
    surviving = list(names)

    for _ in range(max_iterations):
        if len(surviving) <= 1:
            break

        per_fold_deltas = []  # one (n_features,) array per fold: importance_mean (positive = permuting HURT, i.e. feature matters)
        for train_idx, val_idx in cv_splits:
            X_train, X_val = X.iloc[train_idx][surviving], X.iloc[val_idx][surviving]
            y_train, y_val = np.asarray(y)[train_idx], np.asarray(y)[val_idx]

            model = estimator_factory()
            model.fit(X_train, y_train)
            result = permutation_importance(model, X_val, y_val, scoring=sk_scorer, n_repeats=n_repeats, random_state=int(rng.integers(0, 2**31 - 1)))
            per_fold_deltas.append(result.importances_mean)

        deltas = np.stack(per_fold_deltas, axis=0)  # (n_folds, n_features)
        # "Permuting did NOT improve the metric" means importances_mean >= 0 in that fold (permuting a
        # column HURT or was neutral -- sklearn's importance is baseline_score - permuted_score, so a
        # POSITIVE value means permuting made things worse, i.e. the feature is genuinely useful there).
        # A feature is prune-eligible only if permuting IMPROVED the score (negative importance) in EVERY
        # fold -- unanimous evidence the feature is actively harmful/noise, not just weak on average.
        prune_eligible = np.all(deltas < 0, axis=0)
        if not prune_eligible.any():
            break

        surviving = [name for name, eligible in zip(surviving, prune_eligible) if not eligible]

    return surviving


__all__ = ["unanimous_permutation_prune"]
