"""Multi-temperature cross-quantile-band attention.

Iter 58 mechanism. Extends iter 57 cross-quantile-band attention with multi-resolution temperature sweep:
sharp / medium / soft softmax → 3× richer band-membership signal at no extra centroid cost.

Mechanism (regression):
1. Same as iter 57: compute per-band X centroid μ_band_b and per-band y_mean_b, y_std_b over 5 quintile bands.
2. For each temperature t ∈ {0.3, 1.0, 3.0}: softmax(−||q − μ_band_b||² / t) over 5 bands.
3. Per query, per temperature: 5 attention weights + entropy + agg_y_mean + agg_y_std + best_band_idx.
4. Concat across temperatures → 3 × (n_bands + 4) features (=27 for regression; 18 for binary 2-band).

Why this is structurally additive over iter 57:
- Iter 57 used a single fixed temperature (1.0) — query sees one band-membership distribution.
- Iter 58 sweeps sharp (winner-take-most) + medium (calibrated) + soft (smeared) memberships simultaneously.
- The three views capture different "decisiveness" of band assignment; tree boostings get richer feature breadth without
  redoing centroid computation.

Leakage discipline: identical to iter 57 — band assignments + centroids/y-stats per fold from train-fold rows only.

Cost: trivial — same centroid computation as iter 57, just 3× softmax+aggregate evaluations per query batch.
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
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def compute_multi_temp_band_attention_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_bands: int = 5,
    temps: Sequence[float] = _DEFAULT_TEMPS,
    standardize: bool = True,
    column_prefix: str = "mtqbattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Multi-temperature cross-quantile-band attention features.

    Output: per temperature t, n_bands attention weights + entropy + agg_y_mean + agg_y_std + best_band_idx.
    Total features = len(temps) × (n_bands + 4).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    effective_n_bands = 2 if task == "binary" else n_bands
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
        sq = (diffs**2).sum(axis=-1)  # (n_q, n_bands)
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
                band_tag = f"Q{b+1}" if task == "regression" else ("pos" if b == 0 else "neg")
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
    out: np.ndarray = np.zeros((n_train, total_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("multi_temp_band_attention: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
