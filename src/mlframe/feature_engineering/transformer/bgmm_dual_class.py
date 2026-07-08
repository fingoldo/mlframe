"""Dual-class BGM virtuals: fit BayesianGaussianMixture separately on positives AND negatives, sample virtuals from each, expose distances + log-ratio.

Iter 55 mechanism. Extends iter 41/45 BGM (pos-only virtuals) by adding parametric NEGATIVE virtuals — full both-side BGM-density representation.

Mechanism (binary):
1. Fit BGM_pos on positives (n_components_pos=5) and BGM_neg on negatives (n_components_neg=5).
2. Sample 5×n_pos virtuals from BGM_pos and 5×n_neg virtuals from BGM_neg.
3. Per query: distance to k=1,3,5,10-th nearest pos-virtual, same for neg-virtual, plus signed log-gap per k-scale.
4. Total: 8 + 8 + 4 = 20 features.

Differs from iter 45 multi-scale BGM:
- iter 45: sample POSITIVE virtuals at multiple K, all for positive cloud.
- iter 55: sample BOTH positive and negative virtuals at single K each, expose dual-side density.

Differs from iter 28 denrat:
- iter 28: evaluate KDE log-ratio at query — single scalar density signal.
- iter 55: sample virtuals from class-conditional BGM densities, expose KNN-distance features.

Hypothesis: tree boostings benefit from RICHER NEGATIVE virtual cloud, not just positive augmentation. Negatives are abundant in mammography (~3950) — BGM-sampled negative virtuals might surface negative manifold structure CB's TS encoding misses.

Leakage discipline: BGM_pos and BGM_neg refit per fold from train-fold rows only.

Cost: 2× BGM fit + 2× kNN search ~4-6 sec per fold.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _fit_bgmm_and_sample(X_class: np.ndarray, n_synthetic: int, n_components: int, seed: int) -> np.ndarray:
    """Fit BayesianGaussianMixture on class subset and sample virtuals from posterior."""
    from sklearn.mixture import BayesianGaussianMixture
    n_rows = X_class.shape[0]
    if n_rows < n_components + 1:
        return X_class[np.random.default_rng(seed).integers(0, n_rows, size=n_synthetic)].copy().astype(np.float32)
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=200,
        random_state=seed,
        reg_covar=1e-4,
        weight_concentration_prior_type="dirichlet_process",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bgm.fit(X_class)
            samples, _ = bgm.sample(n_synthetic)
        except Exception as exc:
            logger.info("bgmm_dual_class: BGM fit failed (%s); fallback to bootstrap.", exc)
            rng = np.random.default_rng(seed)
            samples = X_class[rng.integers(0, n_rows, size=n_synthetic)]
    return samples.astype(np.float32)


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


def compute_bgmm_dual_class_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_components_pos: int = 5,
    n_components_neg: int = 5,
    oversample_pos: float = 5.0,
    oversample_neg: float = 0.5,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "bdc",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Dual-class BGM virtual distance features.

    Output: 20 columns per row — 4 dist-to-pos-virtual + 4 dist-to-neg-virtual + 4 raw log-gap (real-neg vs virtual-pos) + 4 BGM-virtual log-gap (neg-virtual vs pos-virtual) + 4 mixed-side ratios.
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
            return np.zeros((Xq_s.shape[0], 5 * len(_K_SCALES)), dtype=np.float32)
        k_pos = max(2, min(n_components_pos, Xt_pos.shape[0] // 3))
        k_neg = max(2, min(n_components_neg, Xt_neg.shape[0] // 3))
        n_synth_pos = max(50, int(Xt_pos.shape[0] * oversample_pos))
        n_synth_neg = max(50, int(Xt_neg.shape[0] * oversample_neg))
        X_synth_pos = _fit_bgmm_and_sample(Xt_pos, n_synthetic=n_synth_pos, n_components=k_pos, seed=fold_seed)
        X_synth_neg = _fit_bgmm_and_sample(Xt_neg, n_synthetic=n_synth_neg, n_components=k_neg, seed=fold_seed + 1)
        # Combined virtual + real per class.
        X_virtual_pos = np.concatenate([Xt_pos, X_synth_pos], axis=0)
        X_virtual_neg = np.concatenate([Xt_neg, X_synth_neg], axis=0)
        pos_d = _kth_nearest_dists(X_virtual_pos, Xq_s, max(_K_SCALES))  # (n_q, 4)
        neg_d_virtual = _kth_nearest_dists(X_virtual_neg, Xq_s, max(_K_SCALES))  # (n_q, 4)
        neg_d_real = _kth_nearest_dists(Xt_neg, Xq_s, max(_K_SCALES))  # (n_q, 4)
        log_gap_realneg = np.log(np.maximum(neg_d_real, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
        log_gap_virtneg = np.log(np.maximum(neg_d_virtual, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
        # Mixed-side ratio: how much the BGMM augmentation shifts the neg-side distance (real-neg vs virtual-neg). Distinct from log_gap_*, which both contrast a neg distance against the SAME pos_d; this contrasts the two neg distances against each other, so it isolates the synthetic-augmentation effect on the negative manifold.
        mixed_ratio = np.log(np.maximum(neg_d_real, 1e-9)) - np.log(np.maximum(neg_d_virtual, 1e-9))
        return np.concatenate([pos_d, neg_d_virtual, log_gap_realneg, log_gap_virtneg, mixed_ratio], axis=1).astype(np.float32)

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_pos_k{k}"] = feats[:, j].astype(dtype, copy=False)
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_virtneg_k{k}"] = feats[:, len(_K_SCALES) + j].astype(dtype, copy=False)
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_loggap_realneg_k{k}"] = feats[:, 2 * len(_K_SCALES) + j].astype(dtype, copy=False)
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_loggap_virtneg_k{k}"] = feats[:, 3 * len(_K_SCALES) + j].astype(dtype, copy=False)
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_mixed_ratio_k{k}"] = feats[:, 4 * len(_K_SCALES) + j].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, 5 * len(_K_SCALES)), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("bgmm_dual_class: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
