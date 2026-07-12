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
    bag_feature_subsample: Optional[float] = None,
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
        Controls the per-bag negative subsample draws (and, when ``bag_feature_subsample`` is set,
        the per-bag column draws too -- same rng, so results change if this feature is toggled).
    bag_feature_subsample
        Opt-in second bagging axis. ``None`` (default) trains every bag on the full column set,
        matching prior behaviour bit-for-bit. A float in ``(0, 1]`` draws that fraction of columns
        (at least 1, without replacement, resampled per bag) in addition to the row undersample --
        useful when extreme-imbalance data also carries many weak/noisy features, since decorrelating
        bags on both rows and columns lowers ensemble variance further than row-only bagging.

    Returns
    -------
    dict
        ``test_pred`` (mean predicted positive-class probability across bags, ``(n_test,)``),
        ``bag_preds`` (list of per-bag ``(n_test,)`` prediction arrays), ``models`` (list of fitted models),
        ``bag_feature_idx`` (list of per-bag column-index arrays used, or ``None`` when
        ``bag_feature_subsample`` is not set).
    """
    if bag_feature_subsample is not None and not (0.0 < bag_feature_subsample <= 1.0):
        raise ValueError("easy_ensemble_fit_predict: bag_feature_subsample must be in (0, 1]")

    y_train = np.asarray(y_train)
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("easy_ensemble_fit_predict: y_train must contain both classes")

    n_negatives_per_bag = min(len(neg_idx), max(1, round(negative_ratio * len(pos_idx))))
    rng = np.random.default_rng(random_state)

    is_frame = hasattr(X_train, "iloc")
    n_features = X_train.shape[1]
    n_features_per_bag = max(1, round(bag_feature_subsample * n_features)) if bag_feature_subsample is not None else n_features

    models: List[Any] = []
    bag_preds: List[np.ndarray] = []
    bag_feature_idx: Optional[List[np.ndarray]] = [] if bag_feature_subsample is not None else None
    for _ in range(n_bags):
        sampled_neg = rng.choice(neg_idx, size=n_negatives_per_bag, replace=False)
        bag_idx = np.concatenate([pos_idx, sampled_neg])
        rng.shuffle(bag_idx)

        X_bag = X_train.iloc[bag_idx] if is_frame else X_train[bag_idx]
        X_bag_test = X_test
        y_bag = y_train[bag_idx]

        if bag_feature_subsample is not None:
            feat_idx = np.sort(rng.choice(n_features, size=n_features_per_bag, replace=False))
            bag_feature_idx.append(feat_idx)  # type: ignore[union-attr]
            X_bag = X_bag.iloc[:, feat_idx] if is_frame else X_bag[:, feat_idx]
            X_bag_test = X_test.iloc[:, feat_idx] if is_frame else X_test[:, feat_idx]

        model = model_factory()
        model.fit(X_bag, y_bag)
        models.append(model)
        bag_preds.append(np.asarray(model.predict_proba(X_bag_test)[:, 1], dtype=np.float64))

    test_pred = np.mean(bag_preds, axis=0)
    return {"test_pred": test_pred, "bag_preds": bag_preds, "models": models, "bag_feature_idx": bag_feature_idx}


__all__ = ["easy_ensemble_fit_predict"]
