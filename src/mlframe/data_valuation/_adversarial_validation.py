"""Adversarial validation: a folk-standard Kaggle diagnostic with a clean non-cooperative-game reading.

Train a classifier to distinguish train rows (label 0) from test rows (label 1); its OOF AUC
quantifies train/test distribution shift (0.5 = indistinguishable, no shift; higher = the two sets are
increasingly separable, i.e. more shift), and its per-train-row P(test-like) identifies which training
rows most resemble the test distribution. Reads as a discriminator in a domain-adaptation game: the
classifier is the "player" trying to tell the two distributions apart, and the resulting density-ratio
weights are the practical, near-zero-implementation-cost payoff -- reweighting training rows by how
test-like they are (importance weighting under covariate shift) without any adversarial TRAINING loop
(see :mod:`_adversarial_reweighting` for the actual minimax game -- this module is deliberately simpler
and standalone).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def adversarial_validation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    model: Optional[Any] = None,
    n_splits: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Label train=0/test=1, fit an OOF classifier on the concatenation, quantify shift + suggest weights.

    ``model``: any sklearn-fit-compatible classifier with ``predict_proba`` (default: xgboost, 200
    trees, depth 4). ``X_train``/``X_test`` accept numpy arrays or pandas DataFrames (same columns).

    Returns a dict with:
        ``auc`` -- OOF AUC of train-vs-test discrimination (0.5 = no detectable shift).
        ``train_test_proba`` -- ``(n_train,)`` OOF ``P(this training row looks test-like)``.
        ``top_shift_features`` -- discriminator feature-importance ranking, first 20 names (or
        column indices as strings if the inputs are unlabeled arrays) -- the features driving the
        shift, if any.
        ``suggested_weights`` -- ``(n_train,)`` density-ratio importance weights
        ``p / (1 - p)`` clipped to ``[0.1, 10]`` (bounds the variance of the importance-weighting
        estimator against near-1 probabilities) and renormalized to mean 1.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    if rng is None:
        rng = np.random.default_rng()
    if model is None:
        from xgboost import XGBClassifier

        model = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss", random_state=int(rng.integers(0, 2**31 - 1)))

    if isinstance(X_train, pd.DataFrame):
        feature_names = list(X_train.columns)
        X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    else:
        X_train_arr = np.asarray(X_train)
        X_test_arr = np.asarray(X_test)
        feature_names = [f"f{i}" for i in range(X_train_arr.shape[1])]
        X_all = np.vstack([X_train_arr, X_test_arr])

    n_train = len(X_train)
    n_test = len(X_test)
    y_domain = np.concatenate([np.zeros(n_train, dtype=np.int64), np.ones(n_test, dtype=np.int64)])

    oof_proba = np.zeros(n_train + n_test, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
    importances = np.zeros(len(feature_names), dtype=np.float64)
    for tr_idx, val_idx in kf.split(X_all):
        from sklearn.base import clone

        fold_model = clone(model)
        if isinstance(X_all, pd.DataFrame):
            fold_model.fit(X_all.iloc[tr_idx], y_domain[tr_idx])
            oof_proba[val_idx] = fold_model.predict_proba(X_all.iloc[val_idx])[:, 1]
        else:
            fold_model.fit(X_all[tr_idx], y_domain[tr_idx])
            oof_proba[val_idx] = fold_model.predict_proba(X_all[val_idx])[:, 1]
        fi = getattr(fold_model, "feature_importances_", None)
        if fi is not None:
            importances += np.asarray(fi, dtype=np.float64)
    importances /= n_splits

    auc = float(roc_auc_score(y_domain, oof_proba))
    train_test_proba = oof_proba[:n_train]

    order = np.argsort(-importances)[:20]
    top_shift_features = [feature_names[i] for i in order]

    p = np.clip(train_test_proba, 1e-6, 1.0 - 1e-6)
    raw_weights = p / (1.0 - p)
    raw_weights = np.clip(raw_weights, 0.1, 10.0)
    suggested_weights = raw_weights / raw_weights.mean()

    return dict(
        auc=auc,
        train_test_proba=train_test_proba,
        top_shift_features=top_shift_features,
        suggested_weights=suggested_weights,
    )


__all__ = ["adversarial_validation"]
