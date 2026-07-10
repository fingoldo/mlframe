"""Adversarial-validation-guided fold construction: pick the train rows most similar to the test set.

Random/stratified CV folds implicitly assume the training distribution is representative of what the model
will be scored against. When train and test are NOT exchangeable (covariate shift, as diagnosed by
:func:`mlframe.reporting.charts.drift.adversarial_auc`), a random validation fold gives an overly optimistic
CV estimate. The fix (a widely-used adversarial-validation pattern): fit a train-vs-test classifier, then use
TRAIN rows the classifier finds most "test-like" (highest predicted is-test probability) as the validation
fold -- a validation set that's actually representative of what the model will be scored on.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np


def build_test_like_validation_fold(
    X_train: Any,
    X_test: Any,
    feature_names: Optional[Sequence[str]] = None,
    val_fraction: float = 0.2,
    n_splits: int = 5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    tuple
        ``(val_idx, train_remainder_idx)`` -- integer row indices into ``X_train``: the selected test-like
        validation fold, and the remaining rows (use for the actual training set).
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    if hasattr(X_train, "to_numpy"):
        cols = list(X_train.columns) if feature_names is None else list(feature_names)
        train_arr = X_train[cols].to_numpy(dtype=np.float64)
        test_arr = X_test[cols].to_numpy(dtype=np.float64)
    else:
        train_arr = np.asarray(X_train, dtype=np.float64)
        test_arr = np.asarray(X_test, dtype=np.float64)

    n_train = train_arr.shape[0]
    n_test = test_arr.shape[0]
    if n_train == 0:
        raise ValueError("build_test_like_validation_fold: X_train is empty")

    union = np.concatenate([train_arr, test_arr], axis=0)
    source_label = np.concatenate([np.zeros(n_train, dtype=np.int64), np.ones(n_test, dtype=np.int64)])

    clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=seed, verbosity=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_is_test_proba = cross_val_predict(clf, union, source_label, cv=cv, method="predict_proba")[:, 1]

    train_proba = oof_is_test_proba[:n_train]
    n_val = max(1, round(val_fraction * n_train))
    val_idx = np.argsort(train_proba)[::-1][:n_val]
    val_mask = np.zeros(n_train, dtype=bool)
    val_mask[val_idx] = True
    train_remainder_idx = np.flatnonzero(~val_mask)

    return np.sort(val_idx), train_remainder_idx


__all__ = ["build_test_like_validation_fold"]
