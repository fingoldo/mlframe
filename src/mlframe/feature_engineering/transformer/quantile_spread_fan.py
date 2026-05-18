"""Quantile-spread fan: 3 LGB with quantile losses {0.1, 0.5, 0.9} → spread + skew + position.

Iter 89 mechanism. Agent C #1 ranked.

For regression: fit 3 LGB regressors with quantile losses at α ∈ {0.1, 0.5, 0.9}.
For binary: fit 3 LGB classifiers with focal-loss-like reweighting (gamma ∈ {0, 2, 5}) via
class_weight or sample_weight proxies.

Per query (regression) emit 5 features:
- q10, q50, q90 (3 quantile predictions)
- spread = q90 - q10 (conditional IQR-ish)
- skew_proxy = (q90 + q10 - 2*q50) / (spread + 1e-9)

For binary emit 5 features: 3 reweighted probabilities + their spread + skew.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_quantile_spread_fan_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    standardize: bool = True,
    column_prefix: str = "qfan",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Quantile-spread fan features. 5 outputs per row."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("quantile_spread_fan requires lightgbm") from exc

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt, Xq, y_t, fold_seed):
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        if task == "binary":
            # 3 classifiers with sample_weight reweighting: gamma=0 (uniform), gamma=2 (positive emphasis), gamma=5 (strong)
            preds = np.zeros((Xq_s.shape[0], 3), dtype=np.float32)
            for i, gamma in enumerate([0.0, 2.0, 5.0]):
                if gamma > 0:
                    p_mean = float(y_t.mean())
                    sw = np.where(y_t > 0.5, (1.0 - p_mean) ** gamma, p_mean ** gamma).astype(np.float32)
                else:
                    sw = None
                m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                       random_state=int(fold_seed) + i, verbose=-1, n_jobs=-1)
                m.fit(Xt_s, y_t.astype(np.int32), sample_weight=sw)
                preds[:, i] = m.predict_proba(Xq_s)[:, 1].astype(np.float32)
            q10, q50, q90 = preds[:, 0], preds[:, 1], preds[:, 2]
        else:
            preds = np.zeros((Xq_s.shape[0], 3), dtype=np.float32)
            for i, alpha in enumerate([0.1, 0.5, 0.9]):
                m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                      objective="quantile", alpha=alpha,
                                      random_state=int(fold_seed) + i, verbose=-1, n_jobs=-1)
                m.fit(Xt_s, y_t)
                preds[:, i] = m.predict(Xq_s).astype(np.float32)
            q10, q50, q90 = preds[:, 0], preds[:, 1], preds[:, 2]
        spread = (q90 - q10).astype(np.float32)
        skew_proxy = ((q90 + q10 - 2.0 * q50) / (spread + 1e-9)).astype(np.float32)
        return np.column_stack([q10, q50, q90, spread, skew_proxy])

    def _make_df(feats):
        cols = {}
        suffix_set = ["q10", "q50", "q90", "spread", "skew_proxy"]
        for i, suff in enumerate(suffix_set):
            cols[f"{column_prefix}_{suff}"] = feats[:, i].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("quantile_spread_fan: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
