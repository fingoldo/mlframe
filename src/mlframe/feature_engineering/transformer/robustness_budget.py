"""Robustness budget: per-row prediction stability under Gaussian noise injection.

Iter 75 mechanism. Adversarial agent's #3 ranked (cheapest).

For each query row: inject N=16 small Gaussian perturbations (σ = 0.05 × per-feature std);
re-predict each perturbed copy with baseline LGB; emit pred_mean / pred_std / pred_range / (binary)
flip_rate.

Captures within-baseline prediction stability under input noise — different axis of uncertainty than
iter 69 (between-baseline disagreement).

NO y_query needed. Leakage-free.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_baseline(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("robustness_budget requires lightgbm") from exc
    if task == "binary":
        m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m.fit(Xt, y_t.astype(np.int32))
        return m, True
    else:
        m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m.fit(Xt, y_t)
        return m, False


def compute_robustness_budget_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_perturbations: int = 16,
    sigma_scale: float = 0.05,
    standardize: bool = True,
    column_prefix: str = "robust",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Per-query prediction stability under N Gaussian noise perturbations.

    Output: 5 features per query — pred_orig, pred_mean, pred_std, pred_range, flip_rate (binary) / 0 (regression).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        model, is_binary = _fit_baseline(Xt_s, y_t, task=task, seed=fold_seed)
        # Per-feature std on train (for noise scale).
        feat_std = Xt_s.std(axis=0) * sigma_scale + 1e-6
        rng = np.random.default_rng(int(fold_seed))
        # Original prediction
        if is_binary:
            pred_orig = model.predict_proba(Xq_s)[:, 1].astype(np.float32)
        else:
            pred_orig = model.predict(Xq_s).astype(np.float32)
        # N perturbations
        n_q = Xq_s.shape[0]
        preds_stack = np.zeros((n_perturbations, n_q), dtype=np.float32)
        for i in range(n_perturbations):
            noise = rng.standard_normal(Xq_s.shape).astype(np.float32) * feat_std[None, :]
            Xq_perturbed = Xq_s + noise
            if is_binary:
                preds_stack[i] = model.predict_proba(Xq_perturbed)[:, 1].astype(np.float32)
            else:
                preds_stack[i] = model.predict(Xq_perturbed).astype(np.float32)
        pred_mean = preds_stack.mean(axis=0).astype(np.float32)
        pred_std = preds_stack.std(axis=0).astype(np.float32) + 1e-9
        pred_range = (preds_stack.max(axis=0) - preds_stack.min(axis=0)).astype(np.float32)
        if is_binary:
            # Flip rate: fraction of perturbations where pred crosses 0.5 relative to original.
            orig_class = (pred_orig > 0.5).astype(np.float32)
            perturbed_class = (preds_stack > 0.5).astype(np.float32)
            flip_rate = (perturbed_class != orig_class[None, :]).mean(axis=0).astype(np.float32)
        else:
            flip_rate = np.zeros(n_q, dtype=np.float32)
        return np.column_stack([pred_orig, pred_mean, pred_std, pred_range, flip_rate])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_pred_orig"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_pred_mean"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_pred_std"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_pred_range"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_flip_rate"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("robustness_budget: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
