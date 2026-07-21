"""Cross-quantile-band attention: softmax(query → band_centroid) routes through y-quintile bands.

Iter 57 mechanism. Hybrid of iter 53 inducing-attention (softmax routing) + iter 56 quantile-bands (y-quintile structure).

Mechanism (regression):
1. Split y into 5 quintile bands.
2. Per band: compute X centroid μ_band_b and per-band y_mean_b, y_std_b.
3. Per query: softmax(−||q − μ_band_b||² / temp) over 5 bands = band-attention weights.
4. Output 5 attention weights + entropy + aggregated y_mean + aggregated y_std + max-weight band index = 9 features.

For binary: 2 bands (pos/neg) → smaller feature count.

Why this is structurally new:
- iter 53 inducing-attention: anchors are K-MEANS centroids (data-driven, target-agnostic).
- iter 57 band-attention: anchors are QUANTILE-BAND centroids (target-aware band structure).
- iter 56 quantile-bands: raw distances per band, no softmax routing.

Captures "query's soft membership across y-quintiles" — a target-aware analog of iter 53.

Leakage discipline: band assignments + per-band centroids/y-stats computed per fold from train-fold rows only.

Cost: trivial — K-means-style centroid computation per band + softmax per query. Sub-second per fold.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    """Temperature-scaled softmax over the last axis, max-subtracted for numerical stability."""
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def compute_quantile_band_attention_features(
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
    column_prefix: str = "qbattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Cross-quantile-band attention features.

    Output: n_bands attention weights + entropy + agg_y_mean + agg_y_std + best_band_idx = n_bands + 4 features.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    effective_n_bands = 2 if task == "binary" else n_bands

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Bin the train fold into y-bands (label split for binary, quantile bands for regression), compute each band's feature centroid and y-mean/std, then attend each query row over the band centroids (negative squared distance as the score) to produce per-band weights, entropy, weighted y-mean/std, and the argmax band."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        if task == "binary":
            masks = [y_t > 0.5, y_t <= 0.5]
        else:
            quantiles = np.quantile(y_t, np.linspace(0.0, 1.0, effective_n_bands + 1))
            masks = []
            for b in range(effective_n_bands):
                if b == 0:
                    masks.append(y_t <= quantiles[b + 1])
                elif b == effective_n_bands - 1:
                    masks.append(y_t > quantiles[b])
                else:
                    masks.append((y_t > quantiles[b]) & (y_t <= quantiles[b + 1]))
        # Compute per-band centroid + y stats.
        band_centroids = np.zeros((effective_n_bands, Xt_s.shape[1]), dtype=np.float32)
        band_y_mean = np.zeros(effective_n_bands, dtype=np.float32)
        band_y_std = np.zeros(effective_n_bands, dtype=np.float32)
        band_empty = np.zeros(effective_n_bands, dtype=bool)
        for b, mask in enumerate(masks):
            X_band = Xt_s[mask]
            y_band = y_t[mask]
            if X_band.shape[0] < 1:
                band_empty[b] = True
                continue
            band_centroids[b] = X_band.mean(axis=0)
            band_y_mean[b] = float(y_band.mean())
            band_y_std[b] = float(y_band.std()) + 1e-9
        # Per-query softmax over band centroids.
        diffs = Xq_s[:, None, :] - band_centroids[None, :, :]  # (n_q, n_bands, d)
        sq = (diffs**2).sum(axis=-1)
        scores = -sq
        if band_empty.any():
            # An empty band's zero-initialised centroid sits at the standardized-space origin -- near the
            # data center, not a neutral "far away" point -- so it would otherwise compete for softmax
            # attention weight with a phantom y_mean=y_std=0 and silently pull agg_y_mean/agg_y_std toward
            # 0 for queries near the center. Mask it out of the softmax entirely.
            scores = scores.copy()
            scores[:, band_empty] = -np.inf
        weights = _softmax(scores, temp=temp)  # (n_q, n_bands)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)
        agg_y_mean = (weights * band_y_mean[None, :]).sum(axis=-1).astype(np.float32)
        agg_y_std = (weights * band_y_std[None, :]).sum(axis=-1).astype(np.float32)
        best_band = weights.argmax(axis=-1).astype(np.float32)
        return np.column_stack([weights, entropy, agg_y_mean, agg_y_std, best_band])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Label the raw band-weight/entropy/agg columns with ``column_prefix`` (band tags are Q1..Qn for regression, pos/neg for binary) and cast to the requested output dtype."""
        cols: dict[str, np.ndarray] = {}
        for b in range(effective_n_bands):
            tag = f"Q{b+1}" if task == "regression" else ("pos" if b == 0 else "neg")
            cols[f"{column_prefix}_w_{tag}"] = feats[:, b].astype(dtype, copy=False)
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
        logger.info("quantile_band_attention: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
