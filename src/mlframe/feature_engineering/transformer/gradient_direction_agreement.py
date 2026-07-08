"""Gradient direction agreement: pairwise cosine similarity of finite-diff gradients across 3 baselines.

Iter 80 mechanism. Adversarial agent's #4 ranked. Compares Jacobians, not scalar outputs.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_3baselines(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("gradient_direction_agreement requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression
    if task == "binary":
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1).fit(Xt, y_t.astype(np.int32))
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1).fit(Xt, y_t.astype(np.int32))
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2).fit(Xt, y_t.astype(np.int32))
        except Exception:
            m3 = None
        is_binary = True
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1).fit(Xt, y_t)
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1).fit(Xt, y_t)
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2).fit(Xt, y_t)
        is_binary = False
    return m1, m2, m3, is_binary


def _predict(model, X: np.ndarray, is_binary: bool) -> np.ndarray:
    if is_binary:
        return model.predict_proba(X)[:, 1].astype(np.float32)
    else:
        return model.predict(X).astype(np.float32)


def _gradient(model, X: np.ndarray, is_binary: bool, eps: float) -> np.ndarray:
    """Finite-difference gradient ∇p w.r.t. each feature. Returns (n, d)."""
    n, d = X.shape
    p_base = _predict(model, X, is_binary)
    grad = np.zeros((n, d), dtype=np.float32)
    # Perturb one column in place + restore from a saved copy, instead of copying the
    # whole (n,d) matrix per column. The probe matrix the model sees is bit-identical
    # (col j holds X[:,j]+eps, all other entries untouched), so gradients are unchanged.
    for j in range(d):
        col = X[:, j].copy()
        X[:, j] = col + eps
        p_plus = _predict(model, X, is_binary)
        X[:, j] = col
        grad[:, j] = (p_plus - p_base) / eps
    return grad


def compute_gradient_direction_agreement_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    eps: float = 0.05,
    standardize: bool = True,
    column_prefix: str = "graddir",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Pairwise cosine similarity of gradients across 3 baselines.

    Output: cos12, cos23, cos13, mean_cos, min_cos = 5 features.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        m1, m2, m3, is_binary = _fit_3baselines(Xt_s, y_t, task=task, seed=fold_seed)
        g1 = _gradient(m1, Xq_s, is_binary, eps)
        g2 = _gradient(m2, Xq_s, is_binary, eps)
        if m3 is not None:
            g3 = _gradient(m3, Xq_s, is_binary, eps)
        else:
            g3 = np.zeros_like(g1)

        def _cos(a, b):
            na = np.sqrt((a * a).sum(axis=1)) + 1e-9
            nb = np.sqrt((b * b).sum(axis=1)) + 1e-9
            return ((a * b).sum(axis=1) / (na * nb)).astype(np.float32)

        cos12 = _cos(g1, g2)
        cos23 = _cos(g2, g3)
        cos13 = _cos(g1, g3)
        mean_cos = ((cos12 + cos23 + cos13) / 3.0).astype(np.float32)
        min_cos = np.minimum(cos12, np.minimum(cos23, cos13)).astype(np.float32)
        return np.column_stack([cos12, cos23, cos13, mean_cos, min_cos])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_cos12"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_cos23"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_cos13"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_mean_cos"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_min_cos"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, n_features_out), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("gradient_direction_agreement: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
