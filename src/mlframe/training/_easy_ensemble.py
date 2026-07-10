"""EasyEnsemble-style bagging over independently-undersampled negative subsets, for extreme class imbalance.

Class-weighting reweights the loss but a single model still sees every negative row once. Under EXTREME
imbalance (link prediction, click prediction, fraud) with cheap/plentiful negatives, an alternative that wins
in practice is EasyEnsemble: train N independent classifiers, each on the FULL positive set plus a distinct
random negative subsample at a fixed ratio, then average their predictions -- every negative gets a chance to
be seen across the bag while each individual fit trains on a balanced, cheap-to-fit subset.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np


def easy_ensemble_fit_predict(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    model_factory: Callable[[], Any],
    n_bags: int = 10,
    negative_ratio: float = 1.0,
    random_state: int = 0,
) -> dict:
    """Fit ``n_bags`` classifiers, each on all positives plus a distinct negative undersample, and average.

    Parameters
    ----------
    X_train, y_train
        Training features/binary target (``y_train`` in ``{0, 1}``, ``1`` = minority/positive class).
    X_test
        Rows to predict on.
    model_factory
        Zero-arg factory returning a fresh sklearn-compatible classifier (``.fit(X, y)`` /
        ``.predict_proba(X)``).
    n_bags
        Number of independently-undersampled models to train.
    negative_ratio
        Negatives sampled per bag as a multiple of the positive count (``1.0`` -> balanced 1:1 bags).
    random_state
        Controls the per-bag negative subsample draws.

    Returns
    -------
    dict
        ``test_pred`` (mean predicted positive-class probability across bags, ``(n_test,)``),
        ``bag_preds`` (list of per-bag ``(n_test,)`` prediction arrays), ``models`` (list of fitted models).
    """
    y_train = np.asarray(y_train)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("easy_ensemble_fit_predict: y_train must contain both classes")

    n_negatives_per_bag = min(len(neg_idx), max(1, round(negative_ratio * len(pos_idx))))
    rng = np.random.default_rng(random_state)

    is_frame = hasattr(X_train, "iloc")
    models: List[Any] = []
    bag_preds: List[np.ndarray] = []
    for _ in range(n_bags):
        sampled_neg = rng.choice(neg_idx, size=n_negatives_per_bag, replace=False)
        bag_idx = np.concatenate([pos_idx, sampled_neg])
        rng.shuffle(bag_idx)

        X_bag = X_train.iloc[bag_idx] if is_frame else X_train[bag_idx]
        y_bag = y_train[bag_idx]

        model = model_factory()
        model.fit(X_bag, y_bag)
        models.append(model)
        bag_preds.append(np.asarray(model.predict_proba(X_test)[:, 1], dtype=np.float64))

    test_pred = np.mean(bag_preds, axis=0)
    return {"test_pred": test_pred, "bag_preds": bag_preds, "models": models}


__all__ = ["easy_ensemble_fit_predict"]
