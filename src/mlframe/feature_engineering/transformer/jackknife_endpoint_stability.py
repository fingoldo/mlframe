"""Jackknife+ prediction interval endpoint stability.

Iter 101 mechanism. Agent B #4 ranked.

Per OOF fold + K=10 jackknife-subsample baselines (drop 5% each); for each query, compute spread
of quantile-prediction endpoints across the K runs. Separate upper- vs lower-endpoint stability.

5 features.
"""
from __future__ import annotations
import logging
from typing import Any, Literal, Optional
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_jackknife_endpoint_stability_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    n_subsamples=10, subsample_drop=0.05, standardize=True, column_prefix="jkep", dtype=np.float32,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("jackknife_endpoint_stability requires lightgbm") from exc
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
        n = Xt_s.shape[0]
        rng = np.random.default_rng(int(fold_seed))
        upper_preds = np.zeros((n_subsamples, Xq_s.shape[0]), dtype=np.float32)
        lower_preds = np.zeros((n_subsamples, Xq_s.shape[0]), dtype=np.float32)
        for k in range(n_subsamples):
            keep_mask = rng.random(n) > subsample_drop
            X_sub = Xt_s[keep_mask]
            y_sub = y_t[keep_mask]
            if task == "binary":
                # Two reweighted classifiers as proxy for upper/lower endpoints
                p_mean = float(y_sub.mean())
                sw_up = np.where(y_sub > 0.5, 1.0, p_mean / (1 - p_mean + 1e-6)).astype(np.float32)
                sw_dn = np.where(y_sub > 0.5, (1 - p_mean) / (p_mean + 1e-6), 1.0).astype(np.float32)
                m_up = lgb.LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=int(fold_seed) + k, verbose=-1, n_jobs=-1).fit(X_sub, y_sub.astype(np.int32), sample_weight=sw_up)
                m_dn = lgb.LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=int(fold_seed) + k + 100, verbose=-1, n_jobs=-1).fit(X_sub, y_sub.astype(np.int32), sample_weight=sw_dn)
                upper_preds[k] = m_up.predict_proba(Xq_s)[:, 1].astype(np.float32)
                lower_preds[k] = m_dn.predict_proba(Xq_s)[:, 1].astype(np.float32)
            else:
                m_up = lgb.LGBMRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, objective="quantile", alpha=0.9, random_state=int(fold_seed) + k, verbose=-1, n_jobs=-1).fit(X_sub, y_sub)
                m_dn = lgb.LGBMRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, objective="quantile", alpha=0.1, random_state=int(fold_seed) + k + 100, verbose=-1, n_jobs=-1).fit(X_sub, y_sub)
                upper_preds[k] = m_up.predict(Xq_s).astype(np.float32)
                lower_preds[k] = m_dn.predict(Xq_s).astype(np.float32)
        upper_std = upper_preds.std(axis=0).astype(np.float32) + 1e-9
        lower_std = lower_preds.std(axis=0).astype(np.float32) + 1e-9
        upper_mean = upper_preds.mean(axis=0).astype(np.float32)
        lower_mean = lower_preds.mean(axis=0).astype(np.float32)
        interval_width = (upper_mean - lower_mean).astype(np.float32)
        return np.column_stack([upper_std, lower_std, upper_mean, lower_mean, interval_width])

    def _make_df(feats):
        cols = {}
        cols[f"{column_prefix}_upper_std"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_lower_std"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_upper_mean"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_lower_mean"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_interval_width"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        return pl.DataFrame(_make_df(_process(X_train_f, Xq, y_train_f, seed)))
    if splitter is None:
        raise ValueError("Mode A requires splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_f)):
        out[val_idx] = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100).astype(dtype, copy=False)
        logger.info("jackknife_endpoint_stability: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
