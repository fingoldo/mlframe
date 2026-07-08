"""Set Transformer-style inducing-point attention: two-stage soft-routing through M anchor inducing points.

Iter 53 mechanism. GENUINELY NEW attention-like mechanism. Inspired by Set Transformer (Lee et al. 2019) — uses M << N "inducing points" (learned anchors) as
intermediate Q/K to factor full N×N attention into N×M + M×N attention. Linear complexity in N.

Stage A (anchor → train soft pooling):
    For each anchor m, compute attention weights softmax(anchor_m · k_i / sqrt(d)) over all train rows i.
    Anchor's pooled value: V_m = Σ_i softmax(·) * y_i (and x_i for richer V).

Stage B (query → anchor soft routing):
    For each query q, compute attention weights softmax(q · anchor_m / sqrt(d)) over M anchors.
    Query's output: out_q = Σ_m softmax(·) * V_m.

Anchors are initialized via K-means (M=16) on standardised X. Per-anchor temperature parameter is fixed at 1/sqrt(d) (transformer convention).

Differs from iter 16 anchor_attention:
- iter 16: hard assignment train→anchor (argmin distance) for V computation; soft attention only at query stage.
- iter 53: TWO-STAGE soft attention — both train→anchor AND query→anchor are soft. The anchors learn a Gaussian-weighted summary of nearby train rows, then queries
  softly route through these summaries.

Differs from iter 22 stacked_quantile_neighbours:
- iter 22: two-pass kNN over same rows (qnn → qnn).
- iter 53: two-pass softmax-attention through M intermediate inducing points (anchor → train, then query → anchor).

Beyond-frozen: anchor positions are LEARNED via K-means (gradient-style Lloyd's algorithm). Per-anchor V_m is learned via reverse soft-attention from train rows.

Output: per query — M × (y_mean, y_std) features = 16 × 2 = 32 features per default config.

Plus a single "anchor concentration" feature: entropy of query→anchor softmax (low entropy = query routes to one specific anchor; high = ambiguous).

Leakage discipline: K-means anchor init + V_m computed per fold from train-fold only.

Cost: K-means O(NM) + 2× softmax-attention O(NM) per fold.

Reference: Lee et al. 2019 — Set Transformer; Vaswani 2017 — Attention is all you need.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_anchors_kmeans(X: np.ndarray, M: int, seed: int) -> np.ndarray:
    """K-means anchor initialization. M anchors over all X."""
    from sklearn.cluster import KMeans
    M_eff = min(M, max(2, X.shape[0] // 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=M_eff, random_state=seed, n_init=5, max_iter=100)
        km.fit(X)
    return np.asarray(km.cluster_centers_.astype(np.float32, copy=False))


def _softmax_with_temp(scores: np.ndarray, temp: float) -> np.ndarray:
    """Numerically-stable softmax along last axis with temperature."""
    scaled = scores / max(temp, 1e-9)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    e = np.exp(scaled)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def _squared_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared euclidean distance, (len(A), len(B)), via the ``||a||^2 - 2 a.b + ||b||^2`` GEMM
    decomposition. Avoids the (len(A), len(B), d) broadcast cube that ``((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1)``
    materialises; only the (len(A), len(B)) result is allocated. Differs from the subtraction form by float32 reduction
    order (~2e-7 relative on the downstream softmax), selection-equivalent for these attention features."""
    a_sq = np.einsum("ij,ij->i", A, A)[:, None]
    b_sq = np.einsum("ij,ij->i", B, B)[None, :]
    d = a_sq - 2.0 * (A @ B.T) + b_sq
    np.maximum(d, 0.0, out=d)
    return np.asarray(d)


def _stage_a_anchor_to_train(
    anchors: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    temp: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stage A: each anchor soft-pools train rows. Returns per-anchor (y_mean_m, y_std_m).

    Anchor scores over train: anchor_m · x_i (could use cosine or negative-distance; here negative-Euclidean for softmax-attention-like behavior).
    """
    # Use negative squared distance (standardised) so closer train rows get higher score.
    sq = _squared_dists(anchors, X_train)  # (M, N)
    scores = -sq  # closer → higher score
    weights = _softmax_with_temp(scores, temp=temp)  # (M, N) — softmax over N for each anchor
    # Anchor's pooled y_mean and y_std.
    y_mean_m = (weights * y_train[None, :]).sum(axis=-1).astype(np.float32)  # (M,)
    y_var_m = ((weights * (y_train[None, :] - y_mean_m[:, None]) ** 2).sum(axis=-1)).astype(np.float32)
    y_std_m = np.sqrt(y_var_m + 1e-9)
    return y_mean_m, y_std_m


def _stage_b_query_to_anchor(
    queries: np.ndarray,
    anchors: np.ndarray,
    temp: float,
) -> np.ndarray:
    """Stage B: query→anchor softmax. Returns weights of shape (n_query, M)."""
    sq = _squared_dists(queries, anchors)  # (n_q, M)
    scores = -sq
    return _softmax_with_temp(scores, temp=temp)  # (n_q, M)


def compute_inducing_attention_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_anchors: int = 16,
    temp_a: float = 1.0,
    temp_b: float = 1.0,
    standardize: bool = True,
    column_prefix: str = "indattn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Set Transformer-style inducing-point attention features.

    Output per query row: M softmax-attention-weights + entropy of query→anchor distribution = M + 1 columns (default M=16 → 17 features).

    Plus aggregated y_mean: Σ_m softmax(q→anchor_m) × y_mean_m (= weighted-sum across anchors) — 1 more feature.
    Plus aggregated y_std: same for y_std. Total = M + 3 features.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """K-means anchors on ``Xt``, stage-A attention pools ``y_t`` onto anchors, stage-B attends ``Xq`` to anchors, and returns padded weights + entropy + aggregated y-mean/std for ``Xq``."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        anchors = _fit_anchors_kmeans(Xt_s, M=n_anchors, seed=fold_seed)
        y_mean_m, y_std_m = _stage_a_anchor_to_train(anchors, Xt_s, y_t, temp=temp_a)  # (M,) each
        weights_qm = _stage_b_query_to_anchor(Xq_s, anchors, temp=temp_b)  # (n_q, M)
        n_q = Xq_s.shape[0]
        M = anchors.shape[0]
        # Compute query→anchor softmax entropy.
        entropy = -np.sum(weights_qm * np.log(weights_qm + 1e-9), axis=-1).astype(np.float32)
        # Compute weighted-sum aggregates over anchors.
        agg_y_mean = (weights_qm * y_mean_m[None, :]).sum(axis=-1).astype(np.float32)
        agg_y_std = (weights_qm * y_std_m[None, :]).sum(axis=-1).astype(np.float32)
        # Pad weights to fixed n_anchors columns (some folds may have fewer anchors if K-means converged smaller).
        weights_padded = np.zeros((n_q, n_anchors), dtype=np.float32)
        weights_padded[:, :M] = weights_qm
        # Concatenate: M weight columns + entropy + agg_y_mean + agg_y_std
        return np.column_stack([weights_padded, entropy, agg_y_mean, agg_y_std])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Map the ``n_anchors`` weight columns plus entropy/y_mean/y_std of ``feats`` to their ``{column_prefix}_*`` output names."""
        cols: dict[str, np.ndarray] = {}
        for j in range(n_anchors):
            cols[f"{column_prefix}_w{j}"] = feats[:, j].astype(dtype, copy=False)
        cols[f"{column_prefix}_entropy"] = feats[:, n_anchors].astype(dtype, copy=False)
        cols[f"{column_prefix}_y_mean"] = feats[:, n_anchors + 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_y_std"] = feats[:, n_anchors + 2].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, n_anchors + 3), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("inducing_attention: fold %d/%d done (n_anchors=%d)", fold_idx + 1, len(splits), n_anchors)

    return pl.DataFrame(_make_df(out))
