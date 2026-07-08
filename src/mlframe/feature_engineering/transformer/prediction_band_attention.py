"""Prediction-quintile band attention: bands by baseline ŷ (regression) or p̂ (binary).

Iter 64 mechanism. Orthogonal to residual-band family (iters 60-63): partitions rows by what the
baseline boosting PREDICTS, not by where it makes mistakes.

Mechanism:
1. Fit 50-iter LightGBM baseline → in-sample predictions ŷ (regression) or p̂ (binary).
2. Partition rows into 5 quintile bands by ŷ (or p̂):
   - Regression: very-low / low / mid / high / very-high predicted y.
   - Binary: very-low p (confident neg) / low / mid (decision boundary) / high / very-high p (confident pos).
3. Per band b: X centroid μ_b + per-band y_mean + per-band y_std + per-band prediction_mean.
4. Per query: softmax(−||q − μ_b||² / temp) over 5 bands.
5. Output: 5 weights + entropy + agg_y_mean + agg_y_std + agg_pred + best_band = 10 features.

Why this is structurally orthogonal to iter 60-63:
- Iter 60-63 bands by residual (|residual| or signed) → "where boosting struggles".
- Iter 64 bands by prediction → "what boosting predicts" / "boosting's own partition of X-space".
- For binary p ≈ 0.5 band captures decision-boundary rows; useful uncertainty signal.

Different from iter 57 (y-quintile bands): iter 57 uses TRUE y for partitioning; iter 64 uses BASELINE
PREDICTED y. The difference matters in OOF mode: at query time, true y is unknown but ŷ is computable.
Iter 64 features are honestly inferable at test time without label leak; iter 57's are too (via baseline
on train fold), but here the baseline-aware partitioning is the explicit design intent.

Leakage discipline: identical to iter 60 — baseline fit on train fold only, in-sample predictions used
for band assignment, no label info leaks via centroids.

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
    """Temperature-scaled softmax over the last axis, numerically stabilized by subtracting the row max before exponentiating."""
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int, n_estimators: int = 50, max_depth: int = 3) -> np.ndarray:
    """Fit a small LightGBM baseline (classifier for ``task="binary"``, regressor otherwise) on ``(Xt, y_t)`` and return its in-sample predictions, used to rank training rows into prediction bands."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("prediction_band_attention requires lightgbm") from exc
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
    return np.asarray(preds)


def compute_prediction_band_attention_features(
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
    column_prefix: str = "predbattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Prediction-quintile band attention features."""
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    effective_n_bands = n_bands

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Fit the baseline on the (optionally scaled) train fold, bucket training rows into ``effective_n_bands`` prediction-quantile bands, then attention-weight each query row against the band centroids to produce per-band weights, entropy, aggregated y-mean/std/pred and the best-matching band."""
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
        quantiles = np.quantile(preds_tr, np.linspace(0.0, 1.0, effective_n_bands + 1))
        masks = []
        for b in range(effective_n_bands):
            if b == 0:
                masks.append(preds_tr <= quantiles[b + 1])
            elif b == effective_n_bands - 1:
                masks.append(preds_tr > quantiles[b])
            else:
                masks.append((preds_tr > quantiles[b]) & (preds_tr <= quantiles[b + 1]))

        band_centroids = np.zeros((effective_n_bands, Xt_s.shape[1]), dtype=np.float32)
        band_y_mean = np.zeros(effective_n_bands, dtype=np.float32)
        band_y_std = np.zeros(effective_n_bands, dtype=np.float32)
        band_pred_mean = np.zeros(effective_n_bands, dtype=np.float32)
        for b, mask in enumerate(masks):
            X_band = Xt_s[mask]
            y_band = y_t[mask]
            p_band = preds_tr[mask]
            if X_band.shape[0] < 1:
                continue
            band_centroids[b] = X_band.mean(axis=0)
            band_y_mean[b] = float(y_band.mean())
            band_y_std[b] = float(y_band.std()) + 1e-9
            band_pred_mean[b] = float(p_band.mean())

        diffs = Xq_s[:, None, :] - band_centroids[None, :, :]
        sq = (diffs**2).sum(axis=-1)
        scores = -sq
        weights = _softmax(scores, temp=temp)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)
        agg_y_mean = (weights * band_y_mean[None, :]).sum(axis=-1).astype(np.float32)
        agg_y_std = (weights * band_y_std[None, :]).sum(axis=-1).astype(np.float32)
        agg_pred = (weights * band_pred_mean[None, :]).sum(axis=-1).astype(np.float32)
        best_band = weights.argmax(axis=-1).astype(np.float32)
        return np.column_stack([weights, entropy, agg_y_mean, agg_y_std, agg_pred, best_band])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Slice the ``_process`` output columns into a name-tagged dict of feature columns (per-band weights P1..Pn plus entropy/y_mean/y_std/pred/best_band), cast to the requested output dtype."""
        cols: dict[str, np.ndarray] = {}
        for b in range(effective_n_bands):
            band_tag = f"P{b+1}"  # P1 = lowest predicted, P5 = highest
            cols[f"{column_prefix}_w_{band_tag}"] = feats[:, b].astype(dtype, copy=False)
        col_idx = effective_n_bands
        cols[f"{column_prefix}_entropy"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_mean"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_std"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_pred"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
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
        logger.info("prediction_band_attention: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
