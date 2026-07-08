"""Band-conditional anchor attention: M sub-anchors per y-quintile band.

Iter 59 mechanism. Hybrid of iter 53 (global K-means anchors, CB PR_AUC record-holder +8.41%) +
iter 57/58 (target-aware band centroids).

Mechanism (regression):
1. Split y into 5 quintile bands (or 2 bands binary pos/neg).
2. Per band b: fit K-means with M=4 anchors on X rows in that band → 5×4 = 20 band-tagged anchors total.
3. Each anchor a carries: position μ_a, parent_band_index b_a, parent-band y_mean, parent-band y_std.
4. Per query: softmax(−||q − μ_a||² / temp) over all 20 anchors.
5. Aggregate two ways:
   - flat: agg_y_mean = Σ_a weight_a × parent_band_y_mean_a (= "smooth band membership via fine spatial anchors")
   - per-band: sum weights within each band → band_mass_b (5 values)
6. Output: 20 attention weights + entropy + flat_agg_y_mean + flat_agg_y_std + 5 band-masses + argmax_anchor + argmax_band
   = 30 features regression / 16 binary (8 anchors).

Why this is structurally new:
- Iter 53: global K-means anchors — fine spatial structure but **target-agnostic**.
- Iter 57: band centroids — target-aware but **only 5 coarse anchors**.
- Iter 58: same as 57 but 3 temperatures — same coarse spatial resolution.
- Iter 59: 20 anchors with **band-context labels** → fine spatial AND target-aware simultaneously.

Captures "in which y-band do my nearest training neighbours live, AND which spatial sub-region of that band".

Leakage discipline: bands + per-band K-means + per-band y-stats computed per fold from train-fold rows only.

Cost: K-means × 5 bands at M=4 anchors ~ trivial (~10ms per fold). Distance matrix 20 anchors × n_q comparable to iter 53/57.
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


def compute_band_conditional_anchor_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    n_bands: int = 5,
    anchors_per_band: int = 4,
    temp: float = 1.0,
    standardize: bool = True,
    column_prefix: str = "bcanc",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Band-conditional anchor attention features.

    Output: n_bands*anchors_per_band attention weights + entropy + flat_y_mean + flat_y_std +
    n_bands band-masses + argmax_anchor + argmax_band = n_anchors + n_bands + 5 features.
    """
    from sklearn.cluster import KMeans

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    effective_n_bands = 2 if task == "binary" else n_bands

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

        # Per-band K-means with M anchors each.
        d = Xt_s.shape[1]
        n_anchors_total = effective_n_bands * anchors_per_band
        all_anchors = np.zeros((n_anchors_total, d), dtype=np.float32)
        anchor_parent_band = np.zeros(n_anchors_total, dtype=np.int32)
        band_y_mean = np.zeros(effective_n_bands, dtype=np.float32)
        band_y_std = np.zeros(effective_n_bands, dtype=np.float32)
        for b, mask in enumerate(masks):
            X_band = Xt_s[mask]
            y_band = y_t[mask]
            if X_band.shape[0] < 1:
                continue
            band_y_mean[b] = float(y_band.mean())
            band_y_std[b] = float(y_band.std()) + 1e-9
            # Need at least anchors_per_band points; otherwise pad with centroid replicas.
            n_clusters_eff = min(anchors_per_band, max(1, X_band.shape[0]))
            if n_clusters_eff == 1 or X_band.shape[0] < anchors_per_band:
                centroid = X_band.mean(axis=0) if X_band.shape[0] >= 1 else np.zeros(d, dtype=np.float32)
                for a_local in range(anchors_per_band):
                    idx = b * anchors_per_band + a_local
                    all_anchors[idx] = centroid
                    anchor_parent_band[idx] = b
            else:
                km = KMeans(
                    n_clusters=anchors_per_band,
                    n_init=4,
                    max_iter=50,
                    random_state=int(fold_seed) + b * 17,
                )
                km.fit(X_band)
                for a_local in range(anchors_per_band):
                    idx = b * anchors_per_band + a_local
                    all_anchors[idx] = km.cluster_centers_[a_local].astype(np.float32)
                    anchor_parent_band[idx] = b

        # Per-query softmax over all anchors.
        diffs = Xq_s[:, None, :] - all_anchors[None, :, :]  # (n_q, n_anchors_total, d)
        sq = (diffs**2).sum(axis=-1)
        scores = -sq
        weights = _softmax(scores, temp=temp)  # (n_q, n_anchors_total)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=-1).astype(np.float32)

        anchor_y_mean = band_y_mean[anchor_parent_band]  # (n_anchors_total,)
        anchor_y_std = band_y_std[anchor_parent_band]
        flat_y_mean = (weights * anchor_y_mean[None, :]).sum(axis=-1).astype(np.float32)
        flat_y_std = (weights * anchor_y_std[None, :]).sum(axis=-1).astype(np.float32)

        # Per-band masses.
        n_q = Xq_s.shape[0]
        band_masses = np.zeros((n_q, effective_n_bands), dtype=np.float32)
        for b in range(effective_n_bands):
            band_idx_mask = anchor_parent_band == b
            band_masses[:, b] = weights[:, band_idx_mask].sum(axis=-1).astype(np.float32)

        argmax_anchor = weights.argmax(axis=-1).astype(np.float32)
        argmax_band = band_masses.argmax(axis=-1).astype(np.float32)

        # Concat: weights | entropy | flat_y_mean | flat_y_std | band_masses | argmax_anchor | argmax_band
        return np.column_stack([
            weights,
            entropy,
            flat_y_mean,
            flat_y_std,
            band_masses,
            argmax_anchor,
            argmax_band,
        ])

    n_anchors_total = effective_n_bands * anchors_per_band
    n_features = n_anchors_total + 1 + 1 + 1 + effective_n_bands + 1 + 1  # weights + 4 scalars + band-masses + 2 argmax

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        # weights[a]
        for a in range(n_anchors_total):
            b = a // anchors_per_band
            band_tag = f"Q{b+1}" if task == "regression" else ("pos" if b == 0 else "neg")
            sub = a % anchors_per_band
            cols[f"{column_prefix}_w_{band_tag}_a{sub}"] = feats[:, a].astype(dtype, copy=False)
        col_idx = n_anchors_total
        cols[f"{column_prefix}_entropy"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_flat_y_mean"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_flat_y_std"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        for b in range(effective_n_bands):
            band_tag = f"Q{b+1}" if task == "regression" else ("pos" if b == 0 else "neg")
            cols[f"{column_prefix}_mass_{band_tag}"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_argmax_anchor"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        cols[f"{column_prefix}_argmax_band"] = feats[:, col_idx].astype(dtype, copy=False); col_idx += 1
        return cols

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
        logger.info("band_conditional_anchor: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
