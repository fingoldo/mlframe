"""Adversarial-validation-guided fold construction: pick the train rows most similar to the test set.

Random/stratified CV folds implicitly assume the training distribution is representative of what the model
will be scored against. When train and test are NOT exchangeable (covariate shift, as diagnosed by
:func:`mlframe.reporting.charts.drift.adversarial_auc`), a random validation fold gives an overly optimistic
CV estimate. The fix (a widely-used adversarial-validation pattern): fit a train-vs-test classifier, then use
TRAIN rows the classifier finds most "test-like" (highest predicted is-test probability) as the validation
fold -- a validation set that's actually representative of what the model will be scored on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _oof_is_test_proba(
    train_arr: np.ndarray,
    test_arr: np.ndarray,
    n_splits: int,
    seed: int,
    need_importance: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fit the OOF train-vs-test classifier on the given feature columns and return (oof_proba, importances).

    ``importances`` (only computed when ``need_importance`` -- the one-shot default path skips this extra fit
    entirely, so it costs nothing when the caller never asked for iterative peel-back) comes from a classifier
    fit on the FULL union (not OOF); it never contributes to the returned probabilities, which stay OOF-honest
    via ``cross_val_predict``.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    n_train = train_arr.shape[0]
    n_test = test_arr.shape[0]
    union = np.concatenate([train_arr, test_arr], axis=0)
    source_label = np.concatenate([np.zeros(n_train, dtype=np.int64), np.ones(n_test, dtype=np.int64)])

    clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=seed, verbosity=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_is_test_proba = cross_val_predict(clf, union, source_label, cv=cv, method="predict_proba")[:, 1]

    importances: Optional[np.ndarray] = None
    if need_importance:
        # importance_type="gain" (not the LGBMClassifier default "split" count): a feature that gives one
        # massive, near-perfectly-separating split ranks low by split COUNT (it's used sparingly) but should
        # rank highest for peel-back purposes -- gain reflects how much it actually drove the classification.
        importance_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=seed, verbosity=-1, importance_type="gain")
        importance_clf.fit(union, source_label)
        importances = importance_clf.feature_importances_

    return oof_is_test_proba, importances


def build_test_like_validation_fold(
    X_train: Any,
    X_test: Any,
    feature_names: Optional[Sequence[str]] = None,
    val_fraction: float = 0.2,
    n_splits: int = 5,
    seed: int = 0,
    n_iterations: int = 1,
    top_k_drop_per_iteration: int = 0,
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]]:
    """Select the train rows most similar to the test distribution as a validation fold.

    Parameters
    ----------
    X_train, X_test
        Numeric feature frames/arrays sharing the same columns (encode categoricals to numeric first, same
        contract as :func:`mlframe.reporting.charts.drift.adversarial_auc`'s ``_encode_pair`` path handles
        internally for that function -- this helper expects already-numeric input for a lean, dependency-free
        OOF classifier fit).
    feature_names
        Column names; inferred from ``X_train.columns`` if it's a DataFrame.
    val_fraction
        Fraction of ``X_train`` rows to select as the validation fold (the highest-is-test-probability rows).
    n_splits
        CV folds for the out-of-fold train-vs-test classifier (the "is-test" probability for each train row
        must come from a fold where that row was held out, or the selection would be a self-fulfilling
        in-sample artifact).
    seed
        Controls the CV split and classifier randomness.
    n_iterations
        Opt-in iterative-refinement mode: when > 1, after each classifier fit the top
        ``top_k_drop_per_iteration`` features by classifier importance are dropped and the classifier is
        re-fit on the shrunk feature set, peeling back the layers of train/test drift one at a time. A single
        dominant drift feature (e.g. a collection-date artifact) can swamp the classifier and mask subtler,
        target-relevant covariate shift on the remaining features -- peeling it back lets the fold selection
        surface that shift instead. Default 1 reproduces the original one-shot behavior bit-for-bit.
    top_k_drop_per_iteration
        Number of top-importance features dropped between iterations. Ignored (no peeling) when
        ``n_iterations <= 1``, matching the original one-shot default.
    return_history
        When True, also return the per-iteration AUC-decay history (adversarial AUC of the OOF probabilities
        vs. remaining feature count, plus which features were dropped each round). Default False keeps the
        original 2-tuple return contract unchanged.

    Returns
    -------
    tuple
        ``(val_idx, train_remainder_idx)`` -- integer row indices into ``X_train``: the selected test-like
        validation fold, and the remaining rows (use for the actual training set). When ``return_history`` is
        True, a third element ``history`` (list of per-iteration dicts with keys ``iteration``, ``n_features``,
        ``auc``, ``dropped_features``) is appended.
    """
    from ..metrics.core import fast_roc_auc

    if hasattr(X_train, "to_numpy"):
        cols = list(X_train.columns) if feature_names is None else list(feature_names)
        train_full = X_train[cols].to_numpy(dtype=np.float64)
        test_full = X_test[cols].to_numpy(dtype=np.float64)
    else:
        cols = list(feature_names) if feature_names is not None else [str(i) for i in range(np.asarray(X_train).shape[1])]
        train_full = np.asarray(X_train, dtype=np.float64)
        test_full = np.asarray(X_test, dtype=np.float64)

    n_train = train_full.shape[0]
    n_test = test_full.shape[0]
    if n_train == 0:
        raise ValueError("build_test_like_validation_fold: X_train is empty")

    active_cols = list(range(len(cols)))
    n_effective_iterations = max(1, n_iterations)
    history: List[Dict[str, Any]] = []
    source_label = np.concatenate([np.zeros(n_train, dtype=np.int64), np.ones(n_test, dtype=np.int64)])

    oof_is_test_proba = np.empty(0)
    for it in range(n_effective_iterations):
        train_arr = train_full[:, active_cols]
        test_arr = test_full[:, active_cols]
        is_last_iteration = it == n_effective_iterations - 1
        can_drop = not is_last_iteration and top_k_drop_per_iteration > 0 and len(active_cols) > top_k_drop_per_iteration
        oof_is_test_proba, importances = _oof_is_test_proba(train_arr, test_arr, n_splits, seed, need_importance=can_drop)
        auc = float(fast_roc_auc(source_label, oof_is_test_proba))

        dropped_names: List[str] = []
        if can_drop and importances is not None:
            drop_order = np.argsort(importances)[::-1][:top_k_drop_per_iteration]
            dropped_local = sorted(drop_order.tolist(), reverse=True)
            dropped_names = [cols[active_cols[i]] for i in dropped_local]
            for local_i in dropped_local:
                del active_cols[local_i]

        history.append(
            {
                "iteration": it,
                "n_features": len(active_cols) + len(dropped_names),
                "auc": auc,
                "dropped_features": dropped_names,
            }
        )

        if not dropped_names:
            break

    train_proba = oof_is_test_proba[:n_train]
    n_val = max(1, round(val_fraction * n_train))
    val_idx = np.argsort(train_proba)[::-1][:n_val]
    val_mask = np.zeros(n_train, dtype=bool)
    val_mask[val_idx] = True
    train_remainder_idx = np.flatnonzero(~val_mask)

    if return_history:
        return np.sort(val_idx), train_remainder_idx, history
    return np.sort(val_idx), train_remainder_idx


__all__ = ["build_test_like_validation_fold"]
