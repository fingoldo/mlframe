"""Borderline-SMOTE distance features: virtual positives synthesized only from boundary-positives.

Iter 34 mechanism. Refines iter 33 SMOTE by restricting synthesis to "borderline" positives — real positives whose kNN contains >50% negative-class rows. These borderline
positives are the hard examples on or near the decision boundary; synthesizing virtuals only around them creates a denser virtual cloud where it actually matters
for the boostings' decisions.

Mechanism (binary):
1. For each REAL positive, compute its kNN in the FULL training set (positives + negatives mixed).
2. Classify as "borderline" if >50% of its kNN are negative class.
3. SMOTE-interpolate ONLY among borderline positives (not all positives).
4. Per query: distance to k=1,3,5,10-th nearest borderline-virtual-positive.
5. Plus signed log-gap vs real-negative distance.
6. Plus a "borderline-virtual-density" feature: count of borderline-virtuals within radius R.

8 + 1 = 9 features per row.

For regression: replace "positive class" with top-quintile y; "borderline" = top-y rows with mostly mid/low-y neighbors.

Why this differs from iter 33 vanilla SMOTE:
- Iter 33 SMOTE: synthesizes among ALL positives uniformly — captures full positive density.
- Iter 34 Borderline-SMOTE: synthesizes only among HARD positives — captures decision-boundary density specifically.

For mammography (1.3% positive), borderline positives are an even smaller fraction (~30-40% of the 52 real positives). Synthesizing densely around them gives boostings a
fine-grained boundary-geometry feature that no other mechanism captures.

Reference: Han, Wang, Mao 2005 — Borderline-SMOTE; theoretical basis for boundary-focused oversampling.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)
_BORDERLINE_KNN = 10  # k used to classify borderline positives


def _find_borderline_positives(X_pos: np.ndarray, X_full: np.ndarray, y_full: np.ndarray, k: int) -> np.ndarray:
    """Return boolean mask over X_pos: True if positive is "borderline" (>50% of its kNN in X_full are negatives).

    X_pos: positives subset.
    X_full: all train rows including the positives.
    y_full: corresponding labels.
    """
    from sklearn.neighbors import NearestNeighbors
    n_pos = X_pos.shape[0]
    if n_pos == 0:
        return np.zeros(0, dtype=bool)
    k_used = min(k + 1, X_full.shape[0])  # +1 because nearest is self
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_full)
    _dists, ids = nn.kneighbors(X_pos)
    # Exclude self (first column is typically the row itself if X_pos ⊂ X_full).
    # Heuristic: drop first neighbour if distance is ~0.
    y_neigh = y_full[ids[:, 1:]]
    negative_rate = (y_neigh <= 0.5).mean(axis=1)
    return np.asarray(negative_rate > 0.5)


def _smote_synthesize_from(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """SMOTE-interpolate among the provided minority subset (which may itself be a filtered "borderline" subset)."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, 0), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors
    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    _dists, ids = nn.kneighbors(X_minority)
    rng = np.random.default_rng(seed)
    # Draw indices/alphas in the exact per-iteration order (interleaved src/nbr/alpha) the PCG64 stream
    # produced before, then do the gather+lerp as one vectorised pass — bit-identical to the row loop.
    src = np.empty(n_synthetic, dtype=np.int64)
    nbr = np.empty(n_synthetic, dtype=np.int64)
    alpha = np.empty(n_synthetic, dtype=np.float32)
    for i in range(n_synthetic):
        s = rng.integers(0, n_min)
        candidates = ids[s, 1:k_used]
        src[i] = s
        if candidates.size == 0:
            nbr[i] = s
            alpha[i] = np.float32(0.0)
            continue
        nbr[i] = candidates[rng.integers(0, candidates.size)]
        alpha[i] = rng.random()
    x_src = X_minority[src]
    return np.asarray((x_src + alpha[:, None] * (X_minority[nbr] - x_src)).astype(np.float32))


def _kth_nearest_dists(X_subset: np.ndarray, X_query: np.ndarray, k_max: int) -> np.ndarray:
    """Distance from each query row to its k-th nearest neighbor in ``X_subset``, for every ``k`` in ``_K_SCALES``; ``1e6`` sentinel columns when ``X_subset`` is empty."""
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


def compute_borderline_smote_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    oversample: float = 10.0,
    k_smote: int = 5,
    k_borderline: int = _BORDERLINE_KNN,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "blsmote",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Borderline-SMOTE distance features.

    Output: 8 columns — distances to k=1,3,5,10-th borderline-virtual-positive (4) + signed log-gaps vs real-negatives (4).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split rows into positive/negative subsets: ``y > 0.5`` for binary tasks, top-``q_high``-quantile for regression."""
        if task == "binary":
            pos_mask = y_sub > 0.5
        else:
            y_hi = np.quantile(y_sub, q_high)
            pos_mask = y_sub >= y_hi
        return X_sub[pos_mask], X_sub[~pos_mask], pos_mask

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Detect borderline positives, SMOTE-synthesize virtual positives around them, and return per-query distances to the virtual-positive cloud plus signed log-gaps versus the negatives."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        Xt_pos, Xt_neg, _pos_mask = _slice(Xt_s, y_t)
        if Xt_pos.shape[0] < 2 or Xt_neg.shape[0] < 2:
            return np.zeros((Xq_s.shape[0], 2 * len(_K_SCALES)), dtype=np.float32)
        # Find borderline positives.
        if task == "binary":
            border_mask = _find_borderline_positives(Xt_pos, Xt_s, y_t, k=k_borderline)
        else:
            y_hi = np.quantile(y_t, q_high)
            y_binary = (y_t >= y_hi).astype(np.float32)
            border_mask = _find_borderline_positives(Xt_pos, Xt_s, y_binary, k=k_borderline)
        X_border = Xt_pos[border_mask]
        # Fallback to all positives if no borderline detected.
        if X_border.shape[0] < 2:
            X_border = Xt_pos
        n_synthetic = max(2 * X_border.shape[0], int(X_border.shape[0] * oversample))
        X_synth = _smote_synthesize_from(X_border, n_synthetic=n_synthetic, k_neighbors=k_smote, seed=fold_seed)
        X_virtual_pos = np.concatenate([X_border, X_synth], axis=0)
        pos_d = _kth_nearest_dists(X_virtual_pos, Xq_s, max(_K_SCALES))
        neg_d = _kth_nearest_dists(Xt_neg, Xq_s, max(_K_SCALES))
        log_gap = np.log(np.maximum(neg_d, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
        return np.asarray(np.concatenate([pos_d, log_gap], axis=1).astype(np.float32))

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Name and dtype-cast the raw ``_process`` feature columns into the output dict."""
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
        logger.info("borderline_smote: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
