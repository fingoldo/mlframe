"""Boosting-residual band attention: bands derived from |residual| of 1-iter LightGBM.

Iter 60 mechanism. Structural twist on iter 57/58: replace y-quintile bands with |residual|-quintile bands
from a 1-iter LightGBM (or 50-iter early-stopped) baseline.

Mechanism (regression):
1. Fit a small LightGBM (50 iter, depth 3) on (X_train, y_train) → predictions ŷ on TRAIN rows.
2. Compute |residual| = |y - ŷ| on train rows.
3. Split |residual| into 5 quintile bands → adaptive partitioning by "regions of fitting difficulty":
   - Band Q1: easy rows (boosting fits well)
   - Band Q5: hard rows (large residual — outliers or under-modelled regions)
4. Per band b: compute X centroid μ_band_b + per-band y_mean + per-band y_std.
5. Per query: softmax(−||q − μ_band_b||² / temp) over 5 bands (single temp = 1.0 for v1).
6. Output: 5 attention weights + entropy + agg_y_mean + agg_y_std + best_band_idx = 9 features.

For binary: 1-iter LGB outputs probabilities → residual r = y - p_pred ∈ [-1, 1]; |r| measures classification
confidence. Bands = [well-classified, mildly-borderline, very-borderline, mildly-misclassified, very-misclassified].

Why this is structurally new vs iter 57/58:
- Iter 57/58 used y-MAGNITUDE quintile bands (target-aware but partitions ignore what boosting struggles with).
- Iter 60 uses RESIDUAL-MAGNITUDE bands (target-aware AND boosting-difficulty-aware).
- Bands now represent ADAPTIVE partitioning — the "hard" cluster is data-defined per-fold, not by target rank.

Captures "boosting's blind spots" as features the boosting can then use to localise its own mistakes.

Leakage discipline: 1-iter LGB fit + bands + centroids + y-stats all computed per fold from train-fold rows only.
Predictions on TRAIN rows are from a model fit on those same rows (in-sample) — that's intentional, because the
goal is to measure residual under the LGB hypothesis class, not to forecast unseen test residuals. Per-fold reuse
ensures no test-row label leaks into band assignment.

Cost: 50-iter LGB ~50ms per fold + standard band softmax. Sub-second per fold.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    """Temperature-scaled, numerically stable softmax over the last axis (subtracts the row max before exponentiating)."""
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def _fit_baseline_predict(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int, n_estimators: int = 50, max_depth: int = 3) -> np.ndarray:
    """Fit a small LightGBM on (Xt, y_t), return in-sample predictions on Xt.

    Used solely to compute residuals for adaptive band partitioning — NOT a downstream model.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("residual_band_attention requires lightgbm") from exc
    if task == "binary":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=int(seed),
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(Xt, y_t.astype(np.int32))
        preds = np.asarray(model.predict_proba(Xt))[:, 1].astype(np.float32)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=int(seed),
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(Xt, y_t)
        preds = np.asarray(model.predict(Xt)).astype(np.float32)
    return np.asarray(preds)


def compute_residual_band_attention_features(
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
    column_prefix: str = "rbattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Boosting-residual band attention features.

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
        """Fit the residual-band baseline and centroids on one train fold, then score one query batch against the bands."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        # Baseline LGB fit on standardised train rows.
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
        sq = (diffs**2).sum(axis=-1)
        scores = -sq
        weights = _softmax(scores, temp=temp)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)
        agg_y_mean = (weights * band_y_mean[None, :]).sum(axis=-1).astype(np.float32)
        agg_y_std = (weights * band_y_std[None, :]).sum(axis=-1).astype(np.float32)
        best_band = weights.argmax(axis=-1).astype(np.float32)
        return np.column_stack([weights, entropy, agg_y_mean, agg_y_std, best_band])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Name the flat feature matrix's columns: per-band weights, entropy, aggregate y stats, and best-band index."""
        cols: dict[str, np.ndarray] = {}
        for b in range(effective_n_bands):
            band_tag = f"R{b+1}"  # R1 = easy, R5 = hardest
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
        logger.info("residual_band_attention: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
