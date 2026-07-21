"""Bidirectional residual band attention: |residual| band ASSIGNMENT + per-band SIGNED residual MEAN.

Iter 63 mechanism. Synthesis of iter 60 (|residual| bands, robust structure) and iter 62 (signed
residuals, direction-aware but vulnerable to heavy-tailed targets).

Mechanism:
1. Fit 50-iter LightGBM baseline → in-sample predictions ŷ.
2. Compute |residual| and SIGNED residual.
3. Partition rows into 5 quintile bands by |residual| (iter 60 robust structure).
4. Per band b: compute X centroid μ_b + per-band y_mean + per-band y_std + per-band SIGNED residual mean.
5. Per query: softmax(−||q − μ_b||² / temp) over 5 bands.
6. Output: 5 weights + entropy + agg_y_mean + agg_y_std + agg_SIGNED_residual + best_band = 10 features.

Why this is structurally novel vs iter 60/62:
- Iter 60: |residual| bands, only y-aggregation per band → robust but no direction info.
- Iter 62: signed-residual bands, direction-aware but band geometry collapses on heavy-tailed targets (abalone).
- Iter 63: |residual| band ASSIGNMENT (robust) + per-band SIGNED residual MEAN aggregated as a feature.

The aggregated signed-residual feature tells the downstream boosting: "for this query's top-band, are the
similar training rows being over-predicted (mean signed residual < 0) or under-predicted (mean signed residual > 0)
by the baseline?" — direction information without making band ASSIGNMENT direction-sensitive.

Leakage discipline: identical to iter 60.

Cost: same as iter 60 (50-iter LGB + softmax) plus trivial per-band signed-residual mean.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    """Temperature-scaled softmax over the last axis, numerically stabilized by subtracting the per-row max before exponentiating."""
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int, n_estimators: int = 50, max_depth: int = 3) -> np.ndarray:
    """Fit a shallow LightGBM baseline via an inner KFold(3) and return its OUT-OF-FOLD predictions on Xt.

    An in-sample prediction is close to y_t almost by construction (the model was just fit on these exact
    rows), which systematically understates the true baseline residual and distorts which rows look
    "easy"/"hard" for band assignment. Matches residual_stratified_distance.py's own
    ``_compute_oof_residuals`` pattern in this same cluster. Falls back to a single in-sample fit when there are
    too few rows for a 3-fold inner split.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("bidir_residual_band requires lightgbm") from exc
    n = Xt.shape[0]
    if n < 3:
        if task == "binary":
            model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
            model.fit(Xt, y_t.astype(np.int32))
            return np.asarray(model.predict_proba(Xt))[:, 1].astype(np.float32)
        model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        model.fit(Xt, y_t)
        return np.asarray(model.predict(Xt)).astype(np.float32)

    from sklearn.model_selection import KFold
    preds = np.zeros(n, dtype=np.float32)
    inner_splitter = KFold(n_splits=3, shuffle=True, random_state=int(seed) + 11)
    for inner_idx, (in_tr, in_val) in enumerate(inner_splitter.split(Xt)):
        if task == "binary":
            m = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed) + 7 + inner_idx, verbose=-1, n_jobs=-1)
            m.fit(Xt[in_tr], y_t[in_tr].astype(np.int32))
            preds[in_val] = np.asarray(m.predict_proba(Xt[in_val]))[:, 1].astype(np.float32)
        else:
            m = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed) + 7 + inner_idx, verbose=-1, n_jobs=-1)
            m.fit(Xt[in_tr], y_t[in_tr])
            preds[in_val] = np.asarray(m.predict(Xt[in_val])).astype(np.float32)
    return preds


def compute_bidir_residual_band_features(
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
    column_prefix: str = "bidrbattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Bidirectional residual band attention features.

    Output: n_bands attention weights + entropy + agg_y_mean + agg_y_std + agg_signed_residual + best_band_idx
    = n_bands + 5 features.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    effective_n_bands = n_bands

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Fit the baseline, assign train rows to |residual| quintile bands, compute per-band centroid/y-stats/signed-residual-mean, then attend each query row to the bands via softmax(-squared-distance)."""
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
        signed_residuals = (y_t - preds_tr).astype(np.float32)
        abs_residuals = np.abs(signed_residuals)
        quantiles = np.quantile(abs_residuals, np.linspace(0.0, 1.0, effective_n_bands + 1))
        masks = []
        for b in range(effective_n_bands):
            if b == 0:
                masks.append(abs_residuals <= quantiles[b + 1])
            elif b == effective_n_bands - 1:
                masks.append(abs_residuals > quantiles[b])
            else:
                masks.append((abs_residuals > quantiles[b]) & (abs_residuals <= quantiles[b + 1]))

        band_centroids = np.zeros((effective_n_bands, Xt_s.shape[1]), dtype=np.float32)
        band_y_mean = np.zeros(effective_n_bands, dtype=np.float32)
        band_y_std = np.zeros(effective_n_bands, dtype=np.float32)
        band_signed_residual_mean = np.zeros(effective_n_bands, dtype=np.float32)
        band_empty = np.zeros(effective_n_bands, dtype=bool)
        for b, mask in enumerate(masks):
            X_band = Xt_s[mask]
            y_band = y_t[mask]
            sr_band = signed_residuals[mask]
            if X_band.shape[0] < 1:
                band_empty[b] = True
                continue
            band_centroids[b] = X_band.mean(axis=0)
            band_y_mean[b] = float(y_band.mean())
            band_y_std[b] = float(y_band.std()) + 1e-9
            band_signed_residual_mean[b] = float(sr_band.mean())

        diffs = Xq_s[:, None, :] - band_centroids[None, :, :]
        sq = (diffs**2).sum(axis=-1)
        scores = -sq
        if band_empty.any():
            # Same empty-band-at-origin phantom-anchor issue as quantile_band_attention.py's F2.
            scores = scores.copy()
            scores[:, band_empty] = -np.inf
        weights = _softmax(scores, temp=temp)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)
        agg_y_mean = (weights * band_y_mean[None, :]).sum(axis=-1).astype(np.float32)
        agg_y_std = (weights * band_y_std[None, :]).sum(axis=-1).astype(np.float32)
        agg_signed_residual = (weights * band_signed_residual_mean[None, :]).sum(axis=-1).astype(np.float32)
        best_band = weights.argmax(axis=-1).astype(np.float32)
        return np.column_stack([weights, entropy, agg_y_mean, agg_y_std, agg_signed_residual, best_band])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Split the flat ``_process`` output (per-band weights + aggregate stats) into named columns."""
        cols: dict[str, np.ndarray] = {}
        for b in range(effective_n_bands):
            band_tag = f"R{b+1}"
            cols[f"{column_prefix}_w_{band_tag}"] = feats[:, b].astype(dtype, copy=False)
        col_idx = effective_n_bands
        cols[f"{column_prefix}_entropy"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_mean"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_std"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_signed_resid"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_best_band"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    n_features = effective_n_bands + 5
    out: np.ndarray = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("bidir_residual_band: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
