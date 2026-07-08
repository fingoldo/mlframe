"""Diffusion-noise positive augmentation: virtual positives via learned per-feature Gaussian noise at multiple scales.

Iter 42 mechanism. BEYOND-FROZEN learned (per-feature noise std fit from positives).

Mechanism (binary):
1. Fit per-feature std σ_j on the positive-class subset of X.
2. For each real positive x_pos, generate K virtuals at multiple noise levels α ∈ {0.1, 0.3, 0.5}:
   virtual = x_pos + α × σ ⊙ ε, where ε ~ N(0, I) independent per virtual.
3. Combine real + diffusion-noise virtuals → augmented positive cloud per noise level.
4. Per query: distances to k-th nearest virtual at EACH noise level (3 noise scales × 4 k-scales = 12 features).
5. Plus signed log-gap vs real-negative distance per noise scale.

For regression: replace "positive class" with top-quintile-y.

Why this is structurally different from prior virtual mechanisms:
- iter 33 SMOTE: convex interpolation between positive PAIRS — captures positive manifold via straight lines.
- iter 35 MIXUP: convex interpolation between (positive, negative) pairs — captures decision boundary.
- iter 41 BGM: parametric Gaussian mixture posterior sampling — captures global density.
- iter 42 diffusion: RADIAL Gaussian noise around each individual positive — captures per-positive local-anisotropy.

The "diffusion" framing borrows from diffusion-model literature (Sohl-Dickstein 2015; Ho et al. 2020). At the lowest noise level (α=0.1), virtuals are very close to real positives — captures small perturbations. At α=0.5, virtuals spread into the local manifold — captures broader "positive-region geometry".

Why beyond-frozen: the per-feature noise std σ_j is fit from positive-class data (learned scaling). This is the "small backprop" boundary — we LEARN how much noise each feature can tolerate.

Leakage discipline: σ_j refit per fold from train-fold positives only.

Cost: O(n_pos × K × n_scales) virtual generation + kNN per scale. At n_pos=50, K=10, 3 scales = 1500 virtuals per fold. Fast.

References:
- Sohl-Dickstein et al. 2015 — diffusion probabilistic models.
- Tabular diffusion / score-based generative models (recent literature).
- This is the simplest analog: forward-noise only, no learned denoising.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)
_NOISE_SCALES = (0.1, 0.3, 0.5)


def _diffusion_synthesize(X_pos: np.ndarray, n_virtuals_per_pos: int, noise_scale: float, sigma_per_feature: np.ndarray, seed: int) -> np.ndarray:
    """Generate virtuals = X_pos[i] + noise_scale * sigma_per_feature ⊙ N(0,I) per virtual."""
    n_pos, d = X_pos.shape
    rng = np.random.default_rng(seed)
    n_synthetic = n_pos * n_virtuals_per_pos
    if n_synthetic <= 0:
        return np.zeros((0, d), dtype=np.float32)
    # Repeat each positive n_virtuals_per_pos times.
    base = np.repeat(X_pos, n_virtuals_per_pos, axis=0)
    noise = rng.standard_normal((n_synthetic, d)).astype(np.float32) * noise_scale * sigma_per_feature[None, :]
    return np.asarray((base + noise).astype(np.float32))


def _kth_nearest_dists(X_subset: np.ndarray, X_query: np.ndarray, k_max: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    n_sub = X_subset.shape[0]
    if n_sub == 0:
        return np.full((X_query.shape[0], len(_K_SCALES)), 1e6, dtype=np.float32)
    k_request = min(k_max, n_sub)
    nn = NearestNeighbors(n_neighbors=k_request, algorithm="auto", n_jobs=-1).fit(X_subset)
    dists, _ids = nn.kneighbors(X_query)
    out = np.zeros((X_query.shape[0], len(_K_SCALES)), dtype=np.float32)
    for col_idx, k in enumerate(_K_SCALES):
        eff_k = min(k, k_request)
        out[:, col_idx] = dists[:, eff_k - 1]
    return out


def compute_diffusion_noise_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_virtuals_per_pos: int = 10,
    noise_scales: Tuple[float, ...] = _NOISE_SCALES,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "diff",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Diffusion-noise virtual positive distance features at multiple noise scales.

    Output: ``len(noise_scales) * len(_K_SCALES) * 2`` columns per row — for each (noise_scale, k_scale) pair: distance + log_gap.
    Default 3 × 4 × 2 = 24 columns.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if task == "binary":
            pos = y_sub > 0.5
            return X_sub[pos], X_sub[~pos]
        y_hi = np.quantile(y_sub, q_high)
        y_lo = np.quantile(y_sub, 1.0 - q_high)
        return X_sub[y_sub >= y_hi], X_sub[y_sub <= y_lo]

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        Xt_pos, Xt_neg = _slice(Xt_s, y_t)
        if Xt_pos.shape[0] < 2 or Xt_neg.shape[0] < 2:
            return np.zeros((Xq_s.shape[0], 2 * len(noise_scales) * len(_K_SCALES)), dtype=np.float32)
        # Learn per-feature std from positives.
        sigma_per_feature = Xt_pos.std(axis=0).astype(np.float32) + 1e-6
        neg_d = _kth_nearest_dists(Xt_neg, Xq_s, max(_K_SCALES))
        # For each noise scale, generate virtuals and compute distances.
        all_feats = []
        for scale_idx, ns in enumerate(noise_scales):
            X_virtual = _diffusion_synthesize(Xt_pos, n_virtuals_per_pos=n_virtuals_per_pos, noise_scale=ns, sigma_per_feature=sigma_per_feature, seed=fold_seed + scale_idx * 7)
            X_virtual_full = np.concatenate([Xt_pos, X_virtual], axis=0)
            pos_d = _kth_nearest_dists(X_virtual_full, Xq_s, max(_K_SCALES))
            log_gap = np.log(np.maximum(neg_d, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
            all_feats.append(pos_d)
            all_feats.append(log_gap)
        return np.asarray(np.concatenate(all_feats, axis=1).astype(np.float32))

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        col_idx = 0
        for ns in noise_scales:
            ns_tag = f"n{int(ns*100):02d}"
            for k in _K_SCALES:
                cols[f"{column_prefix}_{ns_tag}_pos_k{k}"] = feats[:, col_idx].astype(dtype, copy=False)
                col_idx += 1
            for k in _K_SCALES:
                cols[f"{column_prefix}_{ns_tag}_loggap_k{k}"] = feats[:, col_idx].astype(dtype, copy=False)
                col_idx += 1
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    n_features = 2 * len(noise_scales) * len(_K_SCALES)
    out: np.ndarray = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("diffusion_noise: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
