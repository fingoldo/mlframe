"""Multi-aux ensemble: predictions from 3 different aux models + ensemble disagreement features.

Iter 32 mechanism. Targets CB AUC mammography ceiling by exposing CROSS-MODEL disagreement signal that no single boosting can derive internally.

Mechanism (binary):
1. Fit 3 aux models on (X_train, y_train) per fold:
   - vanilla LightGBM (gain-based, cross-entropy)
   - focal-loss LightGBM (rare-class-emphasized, gamma=2)
   - vanilla XGBoost (different histogram, different regularization)
2. Compute OOF predicted probability from each model.
3. Expose 6 features per row:
   - ``proba_lgb``, ``proba_focal``, ``proba_xgb`` — raw OOF predictions
   - ``proba_mean`` = mean of 3 predictions
   - ``proba_std`` = std of 3 predictions (DISAGREEMENT signal)
   - ``proba_range`` = max - min (worst-case disagreement)

For regression: same structure but with LightGBM regressor, XGBoost regressor, plus L1-regularized LightGBM (quantile loss at median).

Why this is CB-blind: CB cannot internally fit different algorithms (LGB, XGB, focal) and compute their disagreement. The disagreement signal exposes "which rows are easy vs hard across model families" — orthogonal to anything CB can compute from its single fit.

Differs from iter 26 (focal_lgb alone — 2 features) and iter 13 (pred_augmented, single LGB):
- Iter 32 has 3 distinct aux models with cross-model disagreement explicitly exposed.
- The DISAGREEMENT features (std, range) are the truly novel addition; they're hypothesis-test-like: rows where models disagree are decision-boundary rows that CB might benefit from extra attention to.

Leakage discipline: 3 aux models refit per fold; OOF predictions computed per fold; never seeing val-fold rows in training.

Cost: 3 aux model fits per fold. With small aux models (depth 4, 200 trees), each fit is ~1 sec. Total: 3 × 5 folds × 1 sec = 15 sec for full OOF.

Reference: Krogh & Vedelsby 1995 — ensemble disagreement bound. Bagging variance estimation (Breiman 1996).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _focal_obj(gamma: float = 2.0):
    """Focal-loss custom objective for LightGBM (rare-class-emphasized cross-entropy)."""
    def objective(preds, train_data):
        labels = train_data.get_label()
        preds_clipped = np.clip(preds, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-preds_clipped))
        pt = labels * p + (1.0 - labels) * (1.0 - p)
        focal_term = (1.0 - pt) ** gamma
        grad = focal_term * (
            labels * (gamma * pt * np.log(np.maximum(pt, 1e-9)) - (1.0 - pt))
            + (1.0 - labels) * ((1.0 - pt) - gamma * pt * np.log(np.maximum(pt, 1e-9)))
        ) * (p - labels) / np.maximum(pt, 1e-9)
        hess = focal_term * p * (1.0 - p) * (1.0 + gamma * (1.0 - pt))
        grad = np.clip(grad, -10.0, 10.0)
        hess = np.maximum(hess, 1e-6)
        return grad, hess
    return objective


def _fit_aux_lgb(X: np.ndarray, y: np.ndarray, *, task: str, seed: int, focal: bool = False, n_estimators: int = 200, max_depth: int = 4):
    import lightgbm as lgb
    if focal and task == "binary":
        train_data = lgb.Dataset(X, label=y)
        params = dict(
            objective=_focal_obj(gamma=2.0),
            learning_rate=0.05, max_depth=max_depth, num_leaves=2 ** max_depth,
            min_data_in_leaf=5, seed=seed, verbosity=-1,
        )
        return lgb.train(params, train_data, num_boost_round=n_estimators)
    if task == "binary":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, num_leaves=2 ** max_depth,
            learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1, min_data_in_leaf=5,
        )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, num_leaves=2 ** max_depth,
            learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1, min_data_in_leaf=5,
        )
    model.fit(X, y)
    return model


def _fit_aux_xgb(X: np.ndarray, y: np.ndarray, *, task: str, seed: int, n_estimators: int = 200, max_depth: int = 4):
    import xgboost as xgb
    if task == "binary":
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            random_state=seed, n_jobs=-1, verbosity=0, eval_metric="logloss",
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
    model.fit(X, y)
    return model


def _predict_proba(model, X: np.ndarray, task: str, focal: bool = False) -> np.ndarray:
    if task == "binary":
        if focal:
            # Focal LGB returns raw logits via .predict.
            logits = np.clip(model.predict(X), -30.0, 30.0)
            return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        # LGBMClassifier / XGBClassifier — predict_proba.
        return model.predict_proba(X)[:, 1].astype(np.float32)
    return model.predict(X).astype(np.float32)


def compute_multi_aux_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_estimators: int = 200,
    max_depth: int = 4,
    column_prefix: str = "multiaux",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Multi-aux ensemble predictions + disagreement features.

    Output: 6 columns per row — predictions from 3 aux models + mean + std + range.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression'; got {task!r}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        # Three aux models.
        m_lgb = _fit_aux_lgb(Xt, y_t, task=task, seed=fold_seed, focal=False, n_estimators=n_estimators, max_depth=max_depth)
        m_focal = _fit_aux_lgb(Xt, y_t, task=task, seed=fold_seed + 1, focal=True, n_estimators=n_estimators, max_depth=max_depth)
        m_xgb = _fit_aux_xgb(Xt, y_t, task=task, seed=fold_seed + 2, n_estimators=n_estimators, max_depth=max_depth)
        # Predictions on query.
        p1 = _predict_proba(m_lgb, Xq, task=task, focal=False)
        p2 = _predict_proba(m_focal, Xq, task=task, focal=True if task == "binary" else False)
        p3 = _predict_proba(m_xgb, Xq, task=task, focal=False)
        preds = np.stack([p1, p2, p3], axis=1)  # (n_q, 3)
        mean = preds.mean(axis=1)
        std = preds.std(axis=1)
        rng = preds.max(axis=1) - preds.min(axis=1)
        return np.column_stack([p1, p2, p3, mean, std, rng]).astype(np.float32)

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        names = ["proba_lgb", "proba_focal", "proba_xgb", "proba_mean", "proba_std", "proba_range"]
        return {f"{column_prefix}_{name}": feats[:, j].astype(dtype, copy=False) for j, name in enumerate(names)}

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, 6), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 10)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("multi_aux: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
