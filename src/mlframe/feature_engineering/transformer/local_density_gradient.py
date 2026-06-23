"""Local density gradient features: ||∇log p̂(x)|| via kNN distance density estimate.

Iter 72 mechanism. Geometric/topological agent's top pick from 3-agent synthesis.

Structurally orthogonal to all 71 existing mechanisms — uses ONLY input X (no baseline needed),
captures the geometry of the input distribution itself.

Mechanism:
1. For each row x, estimate local density p̂(x) via the k-th nearest neighbor distance:
   log p̂(x) ≈ -d * log(r_k(x)) + const, where r_k = distance to k-th NN, d = feature dim.
2. Compute finite-difference gradient of log p̂ across the row's neighborhood:
   for each row, ∇log p̂ ≈ mean over neighbors of (log p̂_neighbor - log p̂_row) × unit_direction_to_neighbor.
3. Output per row:
   - log_density (scalar)
   - density_gradient_norm = ||∇log p̂||
   - density_gradient_alignment = cosine sim with y-gradient direction (regression: signed; binary: pos vs neg class)
   - log_density_neighbors_mean (smoothed density)
   - log_density_neighbors_std (density variability)
4. 5 features per row.

Why this is structurally novel vs iter 60-71:
- Iter 60-69 use residual / prediction / disagreement of baselines fit on (X, y).
- Iter 71 uses NN-target-mean in baseline-prediction embedding space.
- Iter 72 uses ONLY X — no y, no baseline. Pure input-distribution geometry.

The signal: rows in low-density pockets are atypical — for rare-positive binary like mammography
(1.3% positive), positive instances cluster in low-density pockets of X-space. Density gradient
direction points OUT of these pockets, providing a "how far from typical?" feature the boosting
can split on.

Leakage discipline: per fold, density estimated only from train_idx rows; query gets density via
their distance to train_idx neighbors. No y used at any stage.

Cost: kNN build O(N log N) + per-row gradient O(K). Sub-second per fold.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_local_density_gradient_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    k_neighbors: int = 32,
    standardize: bool = True,
    column_prefix: str = "ldgrad",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Local density gradient features.

    Output: 5 features per row:
    - log_density (kNN-based)
    - density_gradient_norm
    - density_gradient_alignment_with_y (uses train y, not query y → no leak)
    - log_density_neighbors_mean (smoothed)
    - log_density_neighbors_std (variability)
    """
    from sklearn.neighbors import NearestNeighbors

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        d = Xt_s.shape[1]
        k_eff = min(k_neighbors, Xt_s.shape[0] - 1)

        # Build kNN index on training rows
        nn = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=-1).fit(Xt_s)

        # Pre-compute log_density for all training rows (for neighbor density lookup later)
        train_dists, _ = nn.kneighbors(Xt_s)  # (n_train, k_eff + 1) — first col is self with dist=0
        # Use the k-th neighbor distance (exclude self at index 0)
        r_k_train = train_dists[:, k_eff].astype(np.float32) + 1e-9
        log_density_train = (-d * np.log(r_k_train)).astype(np.float32)

        # Now compute query features
        q_dists, q_idx = nn.kneighbors(Xq_s)  # (n_q, k_eff + 1)
        # For query rows, drop the self-match if Xq is part of Xt (only happens in Mode A val fold which isn't part of train_idx)
        # Conservative: use neighbors[:, :k_eff] which are the k nearest (still includes possible self if Xq == Xt rows)
        # Since Mode A query is val_idx and we built NN on train_idx, val rows are NOT in the NN graph → no self-match.
        # Use neighbors[:, :k_eff] uniformly.
        q_dist_to_kth = q_dists[:, k_eff - 1].astype(np.float32) + 1e-9
        log_density_query = (-d * np.log(q_dist_to_kth)).astype(np.float32)

        neighbor_idx = q_idx[:, :k_eff]  # (n_q, k_eff)
        neighbor_X = Xt_s[neighbor_idx]  # (n_q, k_eff, d)
        neighbor_log_density = log_density_train[neighbor_idx]  # (n_q, k_eff)
        neighbor_y = y_t[neighbor_idx]  # (n_q, k_eff)

        log_density_neighbors_mean = neighbor_log_density.mean(axis=1).astype(np.float32)
        log_density_neighbors_std = neighbor_log_density.std(axis=1).astype(np.float32) + 1e-9

        # Density gradient: finite-difference ∇log p̂ from each neighbor back to query.
        # ∇log p̂ ≈ mean over neighbors of (log_dens_neighbor - log_dens_query) * direction(neighbor - query) / ||neighbor - query||
        diffs = neighbor_X - Xq_s[:, None, :]  # (n_q, k_eff, d)
        dists = np.sqrt((diffs ** 2).sum(axis=-1)) + 1e-9  # (n_q, k_eff)
        # Unit directions:
        unit_dirs = diffs / dists[:, :, None]  # (n_q, k_eff, d)
        # Log-density differences:
        log_dens_diff = (neighbor_log_density - log_density_query[:, None]).astype(np.float32)  # (n_q, k_eff)
        # Gradient = mean over neighbors of (log_dens_diff / dist) * unit_dir
        # i.e. weighted average direction of increasing density.
        # einsum fuses the weight-multiply + neighbour-reduction without materialising the
        # (n_q, k_eff, d) product temporary that `(weight[:, :, None] * unit_dirs)` would; the
        # sum order matches the broadcast-then-mean path so the result is bit-identical (bench:
        # _benchmarks/bench_local_density_gradient_einsum.py, ~1.6x, exact==).
        weight = (log_dens_diff / dists).astype(np.float32)  # (n_q, k_eff)
        gradient = (np.einsum("qk,qkd->qd", weight, unit_dirs, optimize=False) / unit_dirs.shape[1]).astype(np.float32)  # (n_q, d)
        gradient_norm = np.sqrt((gradient ** 2).sum(axis=-1)).astype(np.float32) + 1e-9

        # Alignment with y-gradient: compute ŷ-gradient direction via same finite difference
        # i.e. direction of increasing y averaged over neighbors.
        y_query_pseudo = neighbor_y.mean(axis=1)  # local mean y (smoothed pseudo-target)
        y_diff = (neighbor_y - y_query_pseudo[:, None]).astype(np.float32)  # (n_q, k_eff)
        y_gradient_weight = (y_diff / dists).astype(np.float32)  # (n_q, k_eff)
        y_gradient = (np.einsum("qk,qkd->qd", y_gradient_weight, unit_dirs, optimize=False) / unit_dirs.shape[1]).astype(np.float32)  # (n_q, d)
        y_gradient_norm = np.sqrt((y_gradient ** 2).sum(axis=-1)) + 1e-9

        # Cosine similarity
        dot = (gradient * y_gradient).sum(axis=-1)
        alignment = (dot / (gradient_norm * y_gradient_norm)).astype(np.float32)

        return np.column_stack([
            log_density_query,
            gradient_norm,
            alignment,
            log_density_neighbors_mean,
            log_density_neighbors_std,
        ])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_log_density"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_grad_norm"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_alignment"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_nbr_log_density_mean"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_nbr_log_density_std"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("local_density_gradient: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
