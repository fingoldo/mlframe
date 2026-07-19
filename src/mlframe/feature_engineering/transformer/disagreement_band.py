"""Disagreement-band attention: bands by 3-baseline DISAGREEMENT (std of predictions) quintile.

Iter 70 mechanism. Combines iter 60 (residual-band structure) with iter 69 (3-baseline ensemble
disagreement signal). Bands partition rows by structural ambiguity, not fitting error.

Mechanism:
1. Fit 3 baselines: LGB d=3, LGB d=5, Ridge/LogReg.
2. Per train row: compute in-sample predictions (p1, p2, p3) and disagreement = std(p1, p2, p3).
3. Partition train rows into 5 quintile bands by disagreement:
   - Band Q1: all 3 baselines agree (consensus region)
   - Band Q5: max disagreement (structurally ambiguous region)
4. Per band b: X centroid + y_mean + y_std + mean disagreement.
5. Per query: softmax(−||q − μ_b||² / temp) over 5 bands.
6. Output: 5 weights + entropy + agg_y + agg_y_std + agg_disagreement + best_band = 10 features.

Why disagreement-bands differ from iter 60 |residual|-bands:
- Row r has small |residual| but high disagreement → models disagree but average is correct (true ambiguity).
- Row r has large |residual| but low disagreement → all models agree wrongly (systematic bias).
- Disagreement captures structural ambiguity ORTHOGONAL to fitting error.

For binary: rows where LGB classifies high but Ridge classifies low (or vice versa) → boundary-near rows in
EACH model class. Their X-centroid is the boundary region's center across all model classes.

Leakage discipline: identical to iter 60-69.

Cost: 3× baseline fits + std + softmax. Sub-second.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    """Temperature-scaled softmax over the last axis, with a max-subtraction for numerical stability; ``temp`` is floored to avoid divide-by-zero."""
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def _fit_3baselines_in_sample(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int) -> np.ndarray:
    """Fit 3 baselines on (Xt, y_t), return in-sample predictions as (n_train, 3)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("disagreement_band requires lightgbm") from exc
    from sklearn.linear_model import Ridge, LogisticRegression

    preds = np.zeros((Xt.shape[0], 3), dtype=np.float32)
    if task == "binary":
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t.astype(np.int32))
        preds[:, 0] = np.asarray(m1.predict_proba(Xt))[:, 1].astype(np.float32)
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t.astype(np.int32))
        preds[:, 1] = np.asarray(m2.predict_proba(Xt))[:, 1].astype(np.float32)
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2)
            m3.fit(Xt, y_t.astype(np.int32))
            preds[:, 2] = m3.predict_proba(Xt)[:, 1].astype(np.float32)
        except Exception:
            preds[:, 2] = float(y_t.mean())
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        m1.fit(Xt, y_t)
        preds[:, 0] = np.asarray(m1.predict(Xt)).astype(np.float32)
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1)
        m2.fit(Xt, y_t)
        preds[:, 1] = np.asarray(m2.predict(Xt)).astype(np.float32)
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2)
        m3.fit(Xt, y_t)
        preds[:, 2] = m3.predict(Xt).astype(np.float32)
    return preds


def compute_disagreement_band_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_bands: int = 5,
    temp: float = 1.0,
    standardize: bool = True,
    column_prefix: str = "dbattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Disagreement-band attention features.

    Output: n_bands attention weights + entropy + agg_y + agg_y_std + agg_disagreement + best_band
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
        """Standardize, fit the 3 in-sample baselines to get per-row disagreement, bucket train rows into disagreement-quintile bands, then soft-assign query rows to bands (by centroid distance) and aggregate each band's y/std/disagreement into the final feature row."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        preds_tr = _fit_3baselines_in_sample(Xt_s, y_t, task=task, seed=fold_seed)  # (n, 3)
        disagreement = preds_tr.std(axis=1).astype(np.float32)

        quantiles = np.quantile(disagreement, np.linspace(0.0, 1.0, effective_n_bands + 1))
        masks = []
        for b in range(effective_n_bands):
            if b == 0:
                masks.append(disagreement <= quantiles[b + 1])
            elif b == effective_n_bands - 1:
                masks.append(disagreement > quantiles[b])
            else:
                masks.append((disagreement > quantiles[b]) & (disagreement <= quantiles[b + 1]))

        band_centroids = np.zeros((effective_n_bands, Xt_s.shape[1]), dtype=np.float32)
        band_y_mean = np.zeros(effective_n_bands, dtype=np.float32)
        band_y_std = np.zeros(effective_n_bands, dtype=np.float32)
        band_disagreement_mean = np.zeros(effective_n_bands, dtype=np.float32)
        for b, mask in enumerate(masks):
            X_band = Xt_s[mask]
            y_band = y_t[mask]
            d_band = disagreement[mask]
            if X_band.shape[0] < 1:
                continue
            band_centroids[b] = X_band.mean(axis=0)
            band_y_mean[b] = float(y_band.mean())
            band_y_std[b] = float(y_band.std()) + 1e-9
            band_disagreement_mean[b] = float(d_band.mean())

        diffs = Xq_s[:, None, :] - band_centroids[None, :, :]
        sq = (diffs**2).sum(axis=-1)
        scores = -sq
        weights = _softmax(scores, temp=temp)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)
        agg_y = (weights * band_y_mean[None, :]).sum(axis=-1).astype(np.float32)
        agg_y_std = (weights * band_y_std[None, :]).sum(axis=-1).astype(np.float32)
        agg_dis = (weights * band_disagreement_mean[None, :]).sum(axis=-1).astype(np.float32)
        best_band = weights.argmax(axis=-1).astype(np.float32)
        return np.column_stack([weights, entropy, agg_y, agg_y_std, agg_dis, best_band])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Name the flat ``feats`` columns: per-band attention weights ``{prefix}_w_D{b}`` followed by the fixed entropy/y_mean/y_std/disagreement/best_band summary columns, cast to the output ``dtype``."""
        cols: dict[str, np.ndarray] = {}
        for b in range(effective_n_bands):
            band_tag = f"D{b+1}"
            cols[f"{column_prefix}_w_{band_tag}"] = feats[:, b].astype(dtype, copy=False)
        col_idx = effective_n_bands
        cols[f"{column_prefix}_entropy"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_mean"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_y_std"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_disagreement"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_best_band"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        return cols

    n_features = effective_n_bands + 5

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
        logger.info("disagreement_band: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
