"""Performer-style linear attention: RFF kernel approximation of softmax attention.

Iter 54 mechanism. GENUINELY NEW attention-like, continues iter-53's attention factorization direction.

Standard softmax attention: ``out_q = Σ_i softmax(q·k_i / √d) · y_i``. Computational cost O(N²) for full attention; even our kNN-restricted version is O(N·k).
Performer attention (Choromanski et al. 2021): approximate softmax kernel via positive random features:
    exp(q·k) ≈ φ(q)·φ(k)^T  where φ(x) = exp(W^T x − ||x||²/2)
Then attention factorizes:
    out_q ≈ φ(q) · (φ(K)^T · Y) / (φ(q) · Σ_i φ(k_i))
Compute K-side aggregate ONCE over all N training rows; each query is O(M·d) where M = number of random features. Total O(N·M) instead of O(N²).

For frozen feature engineering: each query gets the Performer-approximated kernel-weighted average of y_train. The full-N attention captures GLOBAL structure that
local kNN attention (row_attention) misses.

Mechanism:
1. Draw W ~ N(0, I) ∈ R^(d × M) random feature directions (M = n_features default).
2. Compute φ(x) = exp(W^T x − ||x||²/2) ∈ R^M for each train and query row.
3. Pre-aggregate K-side: A = φ(K)^T · y_train ∈ R^M; B = Σ_i φ(k_i) ∈ R^M (normalizer).
4. Per query: ŷ_q = (φ(q) · A) / (φ(q) · B).
5. Also expose ŷ_q² (second moment) and full kernel similarity to top anchor (max φ(q)·φ(k_i) over k anchors).

Why structurally new:
- iter 0 RFF: produces RFF FEATURES φ(x) — no kernel aggregation.
- iter 53 inducing attention: TWO-STAGE softmax through M anchors — non-linear softmax.
- iter 54 Performer: linear KERNEL aggregation (no softmax) — different factorization.

Output: 4 features per row — y_estimate, y_estimate², kernel_concentration (max kernel sim to any train), normalizer.

Leakage discipline: W matrix fixed per fold; K-side aggregates computed per fold from train-fold only.

Cost: O(N_train · M + N_query · M); M=128 default.

Reference: Choromanski et al. 2021 — Rethinking Attention with Performers.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _performer_features(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Performer positive RFF: φ(x) = exp(W^T x − ||x||²/2).

    Returns (n_rows, M) feature matrix. Numerically stable via shifting (subtract row max).
    """
    # ||x||² per row.
    x_norm_sq = (X**2).sum(axis=1, keepdims=True)  # (n, 1)
    proj = X @ W  # (n, M)
    log_phi = proj - 0.5 * x_norm_sq  # (n, M)
    # Stabilize: subtract row max before exp.
    log_phi -= log_phi.max(axis=1, keepdims=True)
    return np.asarray(np.exp(log_phi).astype(np.float32))


def compute_performer_attention_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    n_features: int = 128,
    standardize: bool = True,
    column_prefix: str = "perfattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Performer-style linear attention features.

    Output: 4 columns per row — kernel-weighted y_estimate, y_estimate², kernel_concentration, normalizer (log).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Core per-fold pipeline: scale, project into Performer random-feature space, aggregate train-side numerator/denominator (linear-attention y estimate), then for each query row compute the kernel-weighted y estimate + its square, a sampled kernel-concentration (max similarity to a 200-row sample), and the log normalizer."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        d_input = Xt_s.shape[1]
        rng = np.random.default_rng(fold_seed)
        # Random Gaussian projection directions, scaled for Performer kernel approximation.
        W = rng.standard_normal((d_input, n_features)).astype(np.float32) / np.sqrt(d_input)
        phi_k = _performer_features(Xt_s, W)  # (n_train, M)
        phi_q = _performer_features(Xq_s, W)  # (n_query, M)
        # Pre-aggregate K-side: A (numerator) and B (denominator).
        A = phi_k.T @ y_t  # (M,)
        B = phi_k.sum(axis=0)  # (M,)
        # Per-query estimate.
        numer = phi_q @ A  # (n_query,)
        denom = phi_q @ B + 1e-9
        y_estimate = (numer / denom).astype(np.float32)
        y_estimate_sq = (y_estimate**2).astype(np.float32)
        # Kernel concentration: max kernel-similarity over a SAMPLE of train rows (full max is O(N) per query but we sample).
        n_sample = min(200, Xt_s.shape[0])
        sample_idx = rng.choice(Xt_s.shape[0], size=n_sample, replace=False)
        sim_sample = phi_q @ phi_k[sample_idx].T  # (n_query, n_sample)
        kernel_conc = sim_sample.max(axis=1).astype(np.float32)
        normalizer_log = np.log(denom).astype(np.float32)
        return np.column_stack([y_estimate, y_estimate_sq, kernel_conc, normalizer_log])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Split the flat ``_process`` output into the 4 named output columns, cast to the requested output ``dtype``."""
        return {
            f"{column_prefix}_y_est": feats[:, 0].astype(dtype, copy=False),
            f"{column_prefix}_y_est_sq": feats[:, 1].astype(dtype, copy=False),
            f"{column_prefix}_kernel_conc": feats[:, 2].astype(dtype, copy=False),
            f"{column_prefix}_log_normalizer": feats[:, 3].astype(dtype, copy=False),
        }

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, 4), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("performer_attention: fold %d/%d done (n_features=%d)", fold_idx + 1, len(splits), n_features)

    return pl.DataFrame(_make_df(out))
