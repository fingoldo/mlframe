"""Signed-residual band attention: bands derived from SIGNED y - ŷ of 1-iter LightGBM.

Iter 62 mechanism. Twist on iter 60 (|residual| bands): use the SIGNED residual to distinguish
overprediction from underprediction. For binary: distinguish false-positives (predicted high, true 0)
from false-negatives (predicted low, true 1) from well-classified rows.

Mechanism:
1. Fit 50-iter LightGBM baseline on (X_train, y_train) → in-sample predictions ŷ.
2. Compute signed residual r = y - ŷ; split r into 5 quintile bands.
   - Band R1: r << 0 (overprediction; for binary: y=0 predicted high → false-positive prone)
   - Band R2: r < 0 (mild overprediction)
   - Band R3: r ≈ 0 (well-fit / well-classified)
   - Band R4: r > 0 (mild underprediction)
   - Band R5: r >> 0 (underprediction; for binary: y=1 predicted low → false-negative prone)
3. Per band b: compute X centroid μ_b + per-band y_mean + per-band y_std.
4. Per query: softmax(−||q − μ_b||² / temp) over 5 bands.
5. Output: 5 weights + entropy + agg_y_mean + agg_y_std + best_band_idx = 9 features.

Why this is structurally novel vs iter 60:
- Iter 60 |residual| bands group together rows boosting struggles with regardless of direction.
- Iter 62 SIGNED bands separate over- from under-prediction → captures error TYPE not just magnitude.
- For binary, this is the FP-vs-FN distinction the boosting can use directly to set per-class shrinkage.

Leakage discipline: identical to iter 60.

Cost: same as iter 60 (50-iter LGB + softmax).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=-1, keepdims=True)


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int, n_estimators: int = 50, max_depth: int = 3) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("signed_residual_band requires lightgbm") from exc
    if task == "binary":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
            random_state=int(seed), verbose=-1, n_jobs=-1,
        )
        model.fit(Xt, y_t.astype(np.int32))
        preds = model.predict_proba(Xt)[:, 1].astype(np.float32)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
            random_state=int(seed), verbose=-1, n_jobs=-1,
        )
        model.fit(Xt, y_t)
        preds = model.predict(Xt).astype(np.float32)
    return preds


def compute_signed_residual_band_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_bands: int = 5,
    temp: float = 1.0,
    baseline_n_estimators: int = 50,
    baseline_max_depth: int = 3,
    standardize: bool = True,
    column_prefix: str = "srbattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Signed-residual band attention features.

    Output: n_bands attention weights + entropy + agg_y_mean + agg_y_std + best_band_idx = n_bands + 4 features.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    effective_n_bands = n_bands

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        preds_tr = _fit_baseline_predict(
            Xt_s, y_t, task=task, seed=fold_seed,
            n_estimators=baseline_n_estimators, max_depth=baseline_max_depth,
        )
        signed_residuals = (y_t - preds_tr).astype(np.float32)  # NOT abs
        quantiles = np.quantile(signed_residuals, np.linspace(0.0, 1.0, effective_n_bands + 1))
        masks = []
        for b in range(effective_n_bands):
            if b == 0:
                masks.append(signed_residuals <= quantiles[b + 1])
            elif b == effective_n_bands - 1:
                masks.append(signed_residuals > quantiles[b])
            else:
                masks.append((signed_residuals > quantiles[b]) & (signed_residuals <= quantiles[b + 1]))

        band_centroids = np.zeros((effective_n_bands, Xt_s.shape[1]), dtype=np.float32)
        band_y_mean = np.zeros(effective_n_bands, dtype=np.float32)
        band_y_std = np.zeros(effective_n_bands, dtype=np.float32)
        for b, mask in enumerate(masks):
            X_band = Xt_s[mask]
            y_band = y_t[mask]
            if X_band.shape[0] < 1:
                continue
            band_centroids[b] = X_band.mean(axis=0)
            band_y_mean[b] = float(y_band.mean())
            band_y_std[b] = float(y_band.std()) + 1e-9

        diffs = Xq_s[:, None, :] - band_centroids[None, :, :]
        sq = (diffs**2).sum(axis=-1)
        scores = -sq
        weights = _softmax(scores, temp=temp)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)
        agg_y_mean = (weights * band_y_mean[None, :]).sum(axis=-1).astype(np.float32)
        agg_y_std = (weights * band_y_std[None, :]).sum(axis=-1).astype(np.float32)
        best_band = weights.argmax(axis=-1).astype(np.float32)
        return np.column_stack([weights, entropy, agg_y_mean, agg_y_std, best_band])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        # SR1..SR5: signed-residual bands, lowest (most negative / over-prediction) → highest (most positive / under-prediction)
        for b in range(effective_n_bands):
            band_tag = f"SR{b+1}"
            cols[f"{column_prefix}_w_{band_tag}"] = feats[:, b].astype(dtype, copy=False)
        cols[f"{column_prefix}_entropy"] = feats[:, effective_n_bands].astype(dtype, copy=False)
        cols[f"{column_prefix}_y_mean"] = feats[:, effective_n_bands + 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_y_std"] = feats[:, effective_n_bands + 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_best_band"] = feats[:, effective_n_bands + 3].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    n_features = effective_n_bands + 4
    out: np.ndarray = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("signed_residual_band: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
