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
    min_fold_agreement_fraction: Optional[float] = None,
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
    min_fold_agreement_fraction
        Opt-in relaxation of the strict-unanimity rule. ``None`` (default) requires ALL folds to flag a
        feature as prune-eligible (permuting it improved the score in every fold) -- bit-identical to the
        original behavior. When set to a fraction in ``(0.0, 1.0]``, a feature is pruned once at least
        ``ceil(min_fold_agreement_fraction * n_folds)`` folds agree, letting the caller dial between the
        conservative unanimous rule (``1.0``, equivalent to ``None``) and a more aggressive mean-like rule
        (small fractions e.g. ``0.5``) -- useful when a single noisy fold would otherwise block pruning of
        a feature that's genuinely unimportant almost everywhere.

    Returns
    -------
    list of str
        Surviving feature names after iterating to convergence (or ``max_iterations``).
    """
    if min_fold_agreement_fraction is not None and not (0.0 < min_fold_agreement_fraction <= 1.0):
        raise ValueError(f"unanimous_permutation_prune: min_fold_agreement_fraction must be in (0.0, 1.0], got {min_fold_agreement_fraction}")
    import pandas as pd
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer

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
        # When min_fold_agreement_fraction is set, relax this to a k-of-n vote (k = ceil(fraction * n_folds))
        # instead of requiring all n folds -- a strict superset of the unanimous rule at fraction=1.0.
        if min_fold_agreement_fraction is None:
            prune_eligible = np.all(deltas < 0, axis=0)
        else:
            n_folds = deltas.shape[0]
            required_votes = int(np.ceil(min_fold_agreement_fraction * n_folds))
            prune_eligible = np.sum(deltas < 0, axis=0) >= required_votes
        if not prune_eligible.any():
            break

        surviving = [name for name, eligible in zip(surviving, prune_eligible) if not eligible]

    return surviving


__all__ = ["unanimous_permutation_prune"]
