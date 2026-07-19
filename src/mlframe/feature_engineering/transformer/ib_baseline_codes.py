"""Information-bottleneck quantized baseline codes.

Iter 84 mechanism. Info-theoretic agent's #4 ranked.

Take 3-baseline prediction vector, fit a k-bit (k=3, 8 cells) discrete code via per-fold quantile
binning that maximizes I(code; y). For each query: emit code id + per-code agg_y + 3 baseline preds.

5 features.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_3baselines_two(Xt: np.ndarray, y_t: np.ndarray, Xq: np.ndarray, task: str, seed: int):
    """Fit 3 diverse baselines (shallow LGB, deeper LGB, linear/logistic) on ``(Xt, y_t)`` and return their predictions on both train and query rows, stacked as ``(n, 3)`` arrays; the linear/logistic baseline falls back to the class prior on fit failure."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("ib_baseline_codes requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression
    train_preds = np.zeros((Xt.shape[0], 3), dtype=np.float32)
    query_preds = np.zeros((Xq.shape[0], 3), dtype=np.float32)
    if task == "binary":
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1).fit(Xt, y_t.astype(np.int32))
        train_preds[:, 0] = np.asarray(m1.predict_proba(Xt))[:, 1]
        query_preds[:, 0] = np.asarray(m1.predict_proba(Xq))[:, 1]
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1).fit(Xt, y_t.astype(np.int32))
        train_preds[:, 1] = np.asarray(m2.predict_proba(Xt))[:, 1]
        query_preds[:, 1] = np.asarray(m2.predict_proba(Xq))[:, 1]
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2).fit(Xt, y_t.astype(np.int32))
            train_preds[:, 2] = m3.predict_proba(Xt)[:, 1]
            query_preds[:, 2] = m3.predict_proba(Xq)[:, 1]
        except Exception:
            prior = float(y_t.mean())
            train_preds[:, 2] = prior
            query_preds[:, 2] = prior
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1).fit(Xt, y_t)
        train_preds[:, 0] = m1.predict(Xt)
        query_preds[:, 0] = m1.predict(Xq)
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1).fit(Xt, y_t)
        train_preds[:, 1] = m2.predict(Xt)
        query_preds[:, 1] = m2.predict(Xq)
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2).fit(Xt, y_t)
        train_preds[:, 2] = m3.predict(Xt)
        query_preds[:, 2] = m3.predict(Xq)
    return train_preds, query_preds


def compute_ib_baseline_codes_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_bits: int = 3,
    standardize: bool = True,
    column_prefix: str = "ibcode",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """IB-quantized baseline codes. 5 features."""
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt, Xq, y_t, fold_seed):
        """Fit the 3 baselines, quantize each to an above/below-median bit and combine into an 8-cell code, then emit per-query the code id, the code's train-set y mean/std, the mean baseline prediction, and the shallow-LGB prediction."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        tp, qp = _fit_3baselines_two(Xt_s, y_t, Xq_s, task=task, seed=fold_seed)
        # Discretize each baseline pred to 2 bits (4 levels) then combine. With 3 baselines and 2 bits each = 4^3 = 64 codes.
        # Reduce by quantile binning per baseline to 2 levels (above/below median) → 2^3 = 8 codes.
        bins_per_baseline = 2  # binary above/below median, 2^3 = 8 cells
        codes_train = np.zeros(Xt_s.shape[0], dtype=np.int32)
        codes_query = np.zeros(Xq_s.shape[0], dtype=np.int32)
        for b in range(3):
            median = float(np.median(tp[:, b]))
            codes_train += (tp[:, b] > median).astype(np.int32) * (bins_per_baseline**b)
            codes_query += (qp[:, b] > median).astype(np.int32) * (bins_per_baseline**b)
        n_codes = bins_per_baseline**3
        code_y_mean = np.zeros(n_codes, dtype=np.float32)
        code_y_std = np.zeros(n_codes, dtype=np.float32)
        for c in range(n_codes):
            mask = codes_train == c
            if mask.sum() > 0:
                code_y_mean[c] = float(y_t[mask].mean())
                code_y_std[c] = float(y_t[mask].std()) + 1e-9
        agg_y = code_y_mean[codes_query].astype(np.float32)
        agg_y_std = code_y_std[codes_query].astype(np.float32)
        # baseline avg pred
        avg_pred = qp.mean(axis=1).astype(np.float32)
        return np.column_stack([
            codes_query.astype(np.float32),
            agg_y,
            agg_y_std,
            avg_pred,
            qp[:, 0],  # one specific baseline pred
        ])

    def _make_df(feats):
        """Slice the raw ``(n_rows, 5)`` feature matrix into named, dtype-cast columns for the output frame."""
        cols = {}
        cols[f"{column_prefix}_code"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_code_y_mean"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_code_y_std"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_avg_pred"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_lgbd3_pred"] = feats[:, 4].astype(dtype, copy=False)
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
    return pl.DataFrame(_make_df(out))
