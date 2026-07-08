"""Focal-loss LightGBM predictions as features for the downstream boostings.

Iter 26 mechanism. Motivation: mammography is 1.3% positive. Standard binary-cross-entropy weights the loss uniformly across examples, so the auxiliary LGB used in
rfprox / pred_augmented / etc gets dominated by the easy negatives. Focal loss (Lin et al. 2017) reweights ``(1 - p_t)^γ`` per example, putting MORE weight on hard
(typically minority-class) examples. The aux LGB trained with focal loss should produce predictions that capture rare-positive signal more sharply, which then
feeds CB as a CB-can't-derive-internally feature (CB doesn't have a focal objective).

Output: 2 columns per row — ``focal_proba`` (OOF probability from the focal aux LGB) and ``focal_logit`` (log(p/(1-p)) of the same).

Why ``focal_logit`` is a separate column: logits stretch the [0,1] probability space into (-inf, inf), giving CB's symmetric oblivious trees more split-point
resolution at the extremes (which is where the rare-positive signal lives). Probability features compress to [0,1] and trees waste splits resolving near 0.5.

Mode A (OOF): focal LGB refit per fold; val rows get fold-trained predictions.
Mode B (X_query given): focal LGB fit once on full X_train; query rows predicted.

Focal loss implementation: custom LightGBM objective with γ=2 (default per Lin 2017 paper for object-detection imbalance; reasonable for tabular 1.3% positive).

Reference: Lin et al. 2017 — Focal Loss for Dense Object Detection. Implementation pattern from Iizuka's lightgbm-focal-loss public examples (re-derived here).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _focal_loss_obj(gamma: float):
    """LightGBM custom objective: focal loss (Lin et al. 2017) for binary classification.

    Returns (grad, hess) given preds (raw logits) and labels. gamma=2 is the published default.
    """
    def objective(preds, train_data):
        """LightGBM-conforming ``(grad, hess)`` callable closing over ``gamma``; converts raw logits to
        probabilities via a clipped sigmoid before applying the focal modulating factor."""
        labels = train_data.get_label()
        # Clip raw preds before sigmoid to avoid float overflow in exp(-preds).
        preds_clipped = np.clip(preds, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-preds_clipped))
        # Focal modulating factor.
        pt = labels * p + (1.0 - labels) * (1.0 - p)
        # Gradient: d/dlogit of focal CE.
        focal_term = (1.0 - pt) ** gamma
        # First derivative: gradient of focal loss w.r.t. raw logit.
        # See Lin 2017 supplement; for binary with sigmoid output:
        grad = focal_term * (
            labels * (gamma * pt * np.log(np.maximum(pt, 1e-9)) - (1.0 - pt))
            + (1.0 - labels) * ((1.0 - pt) - gamma * pt * np.log(np.maximum(pt, 1e-9)))
        ) * (p - labels) / np.maximum(pt, 1e-9)
        # Approximate Hessian (diagonal); use BCE-style p*(1-p) scaled by focal term.
        hess = focal_term * p * (1.0 - p) * (1.0 + gamma * (1.0 - pt))
        # Clip to avoid numerical issues.
        grad = np.clip(grad, -10.0, 10.0)
        hess = np.maximum(hess, 1e-6)
        return grad, hess
    return objective


def _fit_focal_lgb(X: np.ndarray, y: np.ndarray, *, n_estimators: int, max_depth: int, gamma: float, seed: int):
    """Fit a LightGBM model with focal-loss objective and return the booster."""
    import lightgbm as lgb
    train_data = lgb.Dataset(X, label=y)
    params = dict(
        objective=_focal_loss_obj(gamma),
        learning_rate=0.05,
        max_depth=max_depth,
        num_leaves=2**max_depth,
        min_data_in_leaf=5,
        seed=seed,
        verbosity=-1,
    )
    booster = lgb.train(params, train_data, num_boost_round=n_estimators)
    return booster


def _predict_proba_logit(booster, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict raw logit and sigmoid probability from a focal-objective booster."""
    logits = booster.predict(X).astype(np.float32)
    # Clip logits before sigmoid to avoid float overflow on extreme values.
    logits_clipped = np.clip(logits, -30.0, 30.0)
    proba = (1.0 / (1.0 + np.exp(-logits_clipped))).astype(np.float32)
    return proba, logits


def compute_focal_lgb_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    gamma: float = 2.0,
    n_estimators: int = 300,
    max_depth: int = 5,
    column_prefix: str = "focal",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Focal-loss LightGBM predictions as features for downstream boostings.

    Output: 2 columns — ``{prefix}_proba`` (sigmoid probability) and ``{prefix}_logit`` (raw logit).

    Mode A: focal LGB refit per fold; val rows predicted.
    Mode B: focal LGB fit once on full X_train.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0; got {gamma}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        booster = _fit_focal_lgb(X_train_f, y_train_f, n_estimators=n_estimators, max_depth=max_depth, gamma=gamma, seed=seed)
        proba, logit = _predict_proba_logit(booster, Xq)
        return pl.DataFrame({
            f"{column_prefix}_proba": proba.astype(dtype, copy=False),
            f"{column_prefix}_logit": logit.astype(dtype, copy=False),
        })

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out_proba: np.ndarray = np.zeros(n_train, dtype=dtype)
    out_logit: np.ndarray = np.zeros(n_train, dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = X_train_f[train_idx]
        X_va = X_train_f[val_idx]
        y_tr = y_train_f[train_idx]
        booster = _fit_focal_lgb(X_tr, y_tr, n_estimators=n_estimators, max_depth=max_depth, gamma=gamma, seed=int(seed) + fold_idx)
        proba, logit = _predict_proba_logit(booster, X_va)
        out_proba[val_idx] = proba.astype(dtype, copy=False)
        out_logit[val_idx] = logit.astype(dtype, copy=False)
        logger.info("focal_lgb: fold %d/%d done (n_train=%d, n_val=%d, gamma=%.1f)", fold_idx + 1, len(splits), len(train_idx), len(val_idx), gamma)

    return pl.DataFrame({
        f"{column_prefix}_proba": out_proba,
        f"{column_prefix}_logit": out_logit,
    })
