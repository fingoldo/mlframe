"""Bayesian Gaussian Mixture virtual positive sampling: learned parametric density for virtual minority-class sampling.

Iter 41 mechanism. BEYOND-FROZEN learned-generative.

Combines the structural advantage of iter 33 SMOTE (virtual-data augmentation, additive information) with learned-density quality (BGM's gradient-trained variational
inference over mixture components). Different from SMOTE's hand-crafted convex interpolation: BGM learns the actual joint Gaussian-mixture distribution of the
minority class, then samples virtuals from the learned distribution.

Mechanism (binary):
1. Per fold, fit sklearn `BayesianGaussianMixture(n_components=k_mix)` on positive-class rows only. BGM uses variational inference (mean-field) — gradient-based
   posterior over component weights.
2. Sample ``n_synthetic`` virtual positives from the fitted BGM.
3. Combine real + virtual positives → "BGM-augmented" positive cloud.
4. Per query: distances to k=1,3,5,10-th nearest BGM-virtual + signed log-gap vs real-negatives.

For regression: BGM fit on top-quintile-y rows (analogue to "positive class").

Why this is structurally different from prior beyond-frozen (NCA / AE):
- NCA / AE re-represent X with a learned linear / nonlinear transform → boostings can't easily exploit (redundant with their own learned partitioning).
- BGM SAMPLES new virtuals from a learned density → ADDS information about positive-class manifold that real positives are too sparse to convey.

Why this complements iter 33 SMOTE:
- SMOTE: virtual = α*pos_i + (1-α)*pos_j (linear interpolation only).
- BGM: virtual ~ p(x | class=+) under a learned GMM (samples from full mixture posterior, can produce virtuals OFF the convex hull of real positives in regions
  where the learned density predicts high probability).

Leakage discipline: BGM refit per fold from train-fold positives only.

Cost: sklearn BayesianGaussianMixture(n_components=5) on 50-300 minority rows at d=6-10 trains in ~1-2 sec per fold. Total: ~5-10 sec OOF.

References:
- Attias 1999, Blei & Jordan 2006 — variational Bayes for mixtures.
- Chawla 2002 — SMOTE.
- This combination (BGM oversampling) is sometimes called "model-based SMOTE" or "MoG-SMOTE" in the imbalanced-learning literature, though not in standard packages.
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


def _fit_bgmm_and_sample(
    X_minority: np.ndarray,
    n_synthetic: int,
    n_components: int,
    seed: int,
) -> np.ndarray:
    """Fit BayesianGaussianMixture on minority X and sample n_synthetic virtuals from the posterior."""
    from sklearn.mixture import BayesianGaussianMixture
    n_min = X_minority.shape[0]
    if n_min < n_components + 1:
        # Fall back: replicate real minority rows (no synthesis possible).
        return np.asarray(X_minority[np.random.default_rng(seed).integers(0, n_min, size=n_synthetic)].copy().astype(np.float32))
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
            bgm.fit(X_minority)
            samples, _ = bgm.sample(n_synthetic)
        except Exception as exc:
            logger.info("bgmm_virtual: BGM fit failed (%s); falling back to bootstrap.", exc)
            rng = np.random.default_rng(seed)
            samples = X_minority[rng.integers(0, n_min, size=n_synthetic)]
    return np.asarray(samples.astype(np.float32))


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


def compute_bgmm_virtual_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_synthetic_multiplier: float = 5.0,
    n_components: int = 5,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "bgmm",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """BGM-sampled virtual positive distance features.

    Output: 8 columns per row — 4 distances to k-th nearest BGM-virtual + 4 signed log-gaps vs real-negatives.
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
            return np.zeros((Xq_s.shape[0], 2 * len(_K_SCALES)), dtype=np.float32)
        # Cap n_components by available data.
        k_mix = max(2, min(n_components, Xt_pos.shape[0] // 3))
        n_synthetic = max(50, int(Xt_pos.shape[0] * n_synthetic_multiplier))
        X_synth_pos = _fit_bgmm_and_sample(Xt_pos, n_synthetic=n_synthetic, n_components=k_mix, seed=fold_seed)
        X_virtual_pos = np.concatenate([Xt_pos, X_synth_pos], axis=0)
        pos_d = _kth_nearest_dists(X_virtual_pos, Xq_s, max(_K_SCALES))
        neg_d = _kth_nearest_dists(Xt_neg, Xq_s, max(_K_SCALES))
        log_gap = np.log(np.maximum(neg_d, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
        return np.asarray(np.concatenate([pos_d, log_gap], axis=1).astype(np.float32))

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_pos_k{k}"] = feats[:, j].astype(dtype, copy=False)
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_loggap_k{k}"] = feats[:, len(_K_SCALES) + j].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, 2 * len(_K_SCALES)), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("bgmm_virtual: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
