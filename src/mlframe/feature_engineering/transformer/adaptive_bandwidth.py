"""Adaptive-bandwidth row-attention: per-query softmax temperature derived from local kNN distances.

Standard row-attention uses a single global ``softmax_temp`` (default 1.0). Adaptive bandwidth scales the temperature per-query:

    softmax_temp(q) = median_distance_to_topk_neighbours(q) / temp_scale

In dense local regions (median distance small), temperature is small → sharper attention concentrates on the truly nearest neighbours.
In sparse regions (median distance large), temperature is large → smoother attention spreads over a wider set.

This is the "balloon estimator" pattern from non-parametric density estimation (Loftsgaarden & Quesenberry 1965): local bandwidth adapts to local density to keep
the effective neighbour-count comparable across regions.

For boostings: in dense regions, sharp attention gives a precise "this is what y looks like at exactly this point in X-space" signal; in sparse regions, smooth
attention gives a "this is what y looks like in this neighbourhood" signal. Boostings can split on either kind, while a fixed-temp attention forces one extreme.

Reference: Loftsgaarden & Quesenberry 1965; Silverman 1986 (adaptive kernel density estimation).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._kernels_njit import row_attention_stage4_adaptive_njit
from ._projection import (
    apply_projection,
    build_importance_weighted_projection,
    build_random_projections,
    build_supervised_projections_pls,
)
from ._row_attention_ann import build_hnsw_index, query_topk
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_adaptive_bandwidth_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    n_heads: int = 4,
    head_dim: int = 8,
    k: int = 32,
    temp_scale: float = 1.0,
    projection: Literal["random", "pls", "importance"] = "pls",
    standardize: bool = True,
    aggregate: tuple[str, ...] = ("y_mean", "y_std"),
    column_prefix: str = "abandw",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Row-attention with per-query softmax temperature derived from local neighbour density.

    Output: polars DataFrame of shape ``(N, n_heads * len(aggregate))``.

    Parameters specific to adaptive bandwidth:
        ``temp_scale``  - divisor for the per-query bandwidth. ``temp_scale=1.0`` → use the median distance directly. Lower values (0.5) sharpen attention,
                          higher (2.0) smooth it. Default 1.0 is a reasonable starting point.

    Other parameters identical to ``compute_row_attention``: projection ("random"/"pls"/"importance"), n_heads, head_dim, k, etc.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
    n_train, d_input = X_train.shape

    # Standardise train; refit per fold for OOF.
    from sklearn.preprocessing import RobustScaler

    def _build_projections(X_fold: np.ndarray, y_fold: np.ndarray, fold_seed: int) -> np.ndarray:
        if projection == "random":
            return build_random_projections(d_input=d_input, n_heads=n_heads, head_dim=head_dim, seed=fold_seed, dtype=dtype)
        if projection == "pls":
            return build_supervised_projections_pls(X=X_fold, y=y_fold, n_heads=n_heads, head_dim=head_dim, seed=fold_seed, dtype=dtype)
        if projection == "importance":
            return build_importance_weighted_projection(X=X_fold, y=y_fold, n_heads=n_heads, head_dim=head_dim, seed=fold_seed, dtype=dtype)
        raise ValueError(f"projection must be 'random', 'pls', or 'importance'; got {projection!r}.")

    def _run_attention(X_anchor: np.ndarray, X_pool: np.ndarray, y_pool: np.ndarray, projections: np.ndarray) -> dict[str, np.ndarray]:
        k_proj_pool = apply_projection(X_pool, projections, l2_normalize=True)  # (n_heads, |pool|, head_dim)
        q_proj_anchor = apply_projection(X_anchor, projections, l2_normalize=True)  # (n_heads, |anchor|, head_dim)
        n_anchor = X_anchor.shape[0]
        outs: dict[str, np.ndarray] = {}
        for h in range(n_heads):
            index = build_hnsw_index(k_proj_pool[h], space="cosine", M=16, ef_construction=100, num_threads=None)
            topk_ids, dists = query_topk(index, q_proj_anchor[h], k=k, ef_search=max(k * 2, 64), num_threads=None)
            # Adaptive bandwidth: median of top-k distances per query, divided by temp_scale.
            # cosine distance is in [0, 2]; we use median to get a stable per-row scale.
            median_d = np.median(dists, axis=1).astype(np.float32)
            # Guard tiny temps (clip at small positive value).
            temps = np.maximum(median_d / float(temp_scale), 1e-6).astype(np.float32)
            y_mean_v = np.empty(n_anchor, dtype=np.float32)
            y_std_v = np.empty(n_anchor, dtype=np.float32)
            x_mean_v = np.empty((n_anchor, head_dim), dtype=np.float32)
            row_attention_stage4_adaptive_njit(
                q_proj_anchor[h], k_proj_pool[h], y_pool.astype(np.float32, copy=False),
                topk_ids, temps, y_mean_v, y_std_v, x_mean_v,
            )
            if "y_mean" in aggregate:
                outs[f"y_mean_h{h}"] = y_mean_v.astype(dtype, copy=False)
            if "y_std" in aggregate:
                outs[f"y_std_h{h}"] = y_std_v.astype(dtype, copy=False)
            if "x_mean" in aggregate:
                outs[f"x_mean_h{h}"] = x_mean_v.astype(dtype, copy=False)
        return outs

    if X_query is None:
        # Mode A: OOF.
        outputs: dict[str, np.ndarray] = {}
        for h in range(n_heads):
            if "y_mean" in aggregate:
                outputs[f"y_mean_h{h}"] = np.zeros(n_train, dtype=dtype)
            if "y_std" in aggregate:
                outputs[f"y_std_h{h}"] = np.zeros(n_train, dtype=dtype)
            if "x_mean" in aggregate:
                outputs[f"x_mean_h{h}"] = np.zeros((n_train, head_dim), dtype=dtype)
        for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_train)):
            X_tr = X_train[tr_idx]
            X_va = X_train[va_idx]
            y_tr = y_train[tr_idx]
            if standardize:
                scaler = RobustScaler().fit(X_tr)
                X_tr_s = scaler.transform(X_tr).astype(dtype, copy=False)
                X_va_s = scaler.transform(X_va).astype(dtype, copy=False)
            else:
                X_tr_s = X_tr.astype(dtype, copy=False)
                X_va_s = X_va.astype(dtype, copy=False)
            projections = _build_projections(X_tr_s, y_tr, fold_seed=seed + fold_idx)
            fold_outs = _run_attention(X_va_s, X_tr_s, y_tr, projections)
            for key, arr in fold_outs.items():
                outputs[key][va_idx] = arr
    else:
        # Mode B: single-pass with full train.
        if standardize:
            scaler = RobustScaler().fit(X_train)
            X_tr_s = scaler.transform(X_train).astype(dtype, copy=False)
            X_q_s = scaler.transform(X_query).astype(dtype, copy=False)
        else:
            X_tr_s = X_train.astype(dtype, copy=False)
            X_q_s = X_query.astype(dtype, copy=False)
        projections = _build_projections(X_tr_s, y_train, fold_seed=seed)
        outputs = _run_attention(X_q_s, X_tr_s, y_train, projections)

    # Build polars frame.
    cols: dict[str, np.ndarray] = {}
    for h in range(n_heads):
        for agg in aggregate:
            key = f"{agg}_h{h}"
            arr = outputs.get(key)
            if arr is None:
                continue
            if arr.ndim == 1:
                cols[f"{column_prefix}_h{h}_{agg}"] = arr
            else:
                for d in range(arr.shape[1]):
                    cols[f"{column_prefix}_h{h}_{agg}_d{d}"] = arr[:, d]
    return pl.DataFrame(cols)
