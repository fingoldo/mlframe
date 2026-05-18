"""Multi-temperature boosting-residual band attention.

Iter 61 mechanism. Hybrid of iter 60 (residual-quintile bands) + iter 58 (multi-temperature softmax).

Mechanism:
1. Fit 50-iter LightGBM baseline on (X_train, y_train) → in-sample predictions ŷ.
2. Compute |residual| = |y - ŷ|; split into 5 quintile bands.
3. Per band b: compute X centroid μ_b + per-band y_mean + per-band y_std.
4. Apply 3 temperatures (sharp 0.3, medium 1.0, soft 3.0) to softmax(−||q − μ_b||²) over 5 bands.
5. Output: 3 × (n_bands + 4) = 27 features (regression) / 18 features (binary).

Iter 60 set the diabetes CB PR_AUC record with single temperature; iter 61 sweeps three temperatures
for multi-resolution adaptive-band membership signal.

Why this is structurally additive over iter 60:
- Iter 60 used a single fixed temp (1.0) → query sees one residual-band membership distribution.
- Iter 61 sweeps sharp + medium + soft simultaneously, like iter 58 over iter 57.
- Three temperatures capture different "decisiveness" of difficulty-band assignment.

Cost: same baseline LGB + 3× softmax+aggregate (each is trivial). Sub-second per fold.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Sequence

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_DEFAULT_TEMPS: tuple[float, ...] = (0.3, 1.0, 3.0)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=-1, keepdims=True)


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int, n_estimators: int = 50, max_depth: int = 3) -> np.ndarray:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("multi_temp_residual_band requires lightgbm") from exc
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


def compute_multi_temp_residual_band_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_bands: int = 5,
    temps: Sequence[float] = _DEFAULT_TEMPS,
    baseline_n_estimators: int = 50,
    baseline_max_depth: int = 3,
    standardize: bool = True,
    column_prefix: str = "mtrbattn",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Multi-temperature boosting-residual band attention features."""
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    effective_n_bands = n_bands
    temps_list = tuple(float(t) for t in temps)
    n_temps = len(temps_list)
    features_per_temp = effective_n_bands + 4

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
        residuals = np.abs(y_t - preds_tr).astype(np.float32)
        quantiles = np.quantile(residuals, np.linspace(0.0, 1.0, effective_n_bands + 1))
        masks = []
        for b in range(effective_n_bands):
            if b == 0:
                masks.append(residuals <= quantiles[b + 1])
            elif b == effective_n_bands - 1:
                masks.append(residuals > quantiles[b])
            else:
                masks.append((residuals > quantiles[b]) & (residuals <= quantiles[b + 1]))

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
        sq = (diffs ** 2).sum(axis=-1)
        scores = -sq
        n_q = Xq_s.shape[0]
        out_blocks = np.zeros((n_q, n_temps * features_per_temp), dtype=np.float32)
        for ti, t in enumerate(temps_list):
            weights = _softmax(scores, temp=t)
            entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)
            agg_y_mean = (weights * band_y_mean[None, :]).sum(axis=-1).astype(np.float32)
            agg_y_std = (weights * band_y_std[None, :]).sum(axis=-1).astype(np.float32)
            best_band = weights.argmax(axis=-1).astype(np.float32)
            base = ti * features_per_temp
            out_blocks[:, base : base + effective_n_bands] = weights
            out_blocks[:, base + effective_n_bands] = entropy
            out_blocks[:, base + effective_n_bands + 1] = agg_y_mean
            out_blocks[:, base + effective_n_bands + 2] = agg_y_std
            out_blocks[:, base + effective_n_bands + 3] = best_band
        return out_blocks

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        for ti, t in enumerate(temps_list):
            base = ti * features_per_temp
            t_tag = f"t{t:g}".replace(".", "p")
            for b in range(effective_n_bands):
                band_tag = f"R{b+1}"
                cols[f"{column_prefix}_{t_tag}_w_{band_tag}"] = feats[:, base + b].astype(dtype, copy=False)
            cols[f"{column_prefix}_{t_tag}_entropy"] = feats[:, base + effective_n_bands].astype(dtype, copy=False)
            cols[f"{column_prefix}_{t_tag}_y_mean"] = feats[:, base + effective_n_bands + 1].astype(dtype, copy=False)
            cols[f"{column_prefix}_{t_tag}_y_std"] = feats[:, base + effective_n_bands + 2].astype(dtype, copy=False)
            cols[f"{column_prefix}_{t_tag}_best_band"] = feats[:, base + effective_n_bands + 3].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    total_features = n_temps * features_per_temp
    out = np.zeros((n_train, total_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("multi_temp_residual_band: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
