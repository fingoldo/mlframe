"""Active virtual placement: SMOTE virtuals filtered to aux LGB's decision-boundary region.

Iter 49 mechanism. BEYOND-FROZEN active-learning analog. Generates SMOTE virtuals liberally, then keeps ONLY those where aux LGB is uncertain
(``|p − 0.5| < margin_threshold``). Captures decision-boundary geometry.

Mechanism (binary):
1. Generate large pool of SMOTE virtuals from positive class.
2. Train aux LGB on (X_train, y_train).
3. Score virtuals with aux LGB; keep only those with predicted probability near 0.5 (boundary-uncertain).
4. Combine real positives + boundary virtuals → "boundary-augmented" positive cloud.
5. Per query: distances to k-th nearest boundary virtual + signed log-gap vs real-negatives.

Why this differs from iter 43 pseudo-SMOTE:
- iter 43: kept virtuals with P(y=1) ≥ 0.7 (HIGH confidence) → virtuals deep in positive region.
- iter 49: keeps virtuals with |P(y=1) − 0.5| < 0.15 (LOW confidence) → virtuals at decision boundary.

The boundary is where the boostings make their critical decisions. Virtuals there might add information that high-confidence virtuals don't.

For regression: train aux LGB regressor, keep virtuals where predicted-y is near the median (boundary-y).

Leakage discipline: aux LGB refit per fold from train-fold rows only.

Cost: SMOTE + aux LGB ~1-2 sec per fold.

References:
- Active learning literature (Settles 2009): uncertainty sampling.
- Adapted to virtual sample placement.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _smote_synthesize_intra(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """SMOTE intra-positive convex interpolation."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, X_minority.shape[1] if n_min > 0 else 1), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors
    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    _dists, ids = nn.kneighbors(X_minority)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_synthetic, X_minority.shape[1]), dtype=np.float32)
    for i in range(n_synthetic):
        src_idx = rng.integers(0, n_min)
        candidates = ids[src_idx, 1:k_used]
        if candidates.size == 0:
            out[i] = X_minority[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
        alpha = rng.random()
        out[i] = X_minority[src_idx] + alpha * (X_minority[nbr_idx] - X_minority[src_idx])
    return out.astype(np.float32)


def _filter_boundary_virtuals(X_train: np.ndarray, y_train: np.ndarray, virtuals: np.ndarray, task: str, seed: int, margin_threshold: float, n_estimators: int = 200, max_depth: int = 4) -> np.ndarray:
    """Train aux LGB, keep virtuals near decision boundary (|p - 0.5| < margin_threshold for binary)."""
    import lightgbm as lgb
    params = dict(
        n_estimators=n_estimators, max_depth=max_depth, num_leaves=2 ** max_depth,
        learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1, min_data_in_leaf=5,
    )
    if task == "binary":
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        proba = np.asarray(model.predict_proba(virtuals))[:, 1]
        keep_mask = np.abs(proba - 0.5) < margin_threshold
    else:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(virtuals))
        y_median = float(np.median(y_train))
        y_std = float(np.std(y_train)) + 1e-9
        # Keep virtuals predicted near median y (the "boundary" for regression).
        keep_mask = np.abs(pred - y_median) < margin_threshold * y_std
    filtered = virtuals[keep_mask]
    if filtered.shape[0] < 10:
        # Fallback: keep top-N most uncertain virtuals.
        # Wave 62 (2026-05-20): lexsort with row-index tiebreak so tied uncertainty
        # (rounded proba/pred values) gives deterministic top-K across runs.
        if task == "binary":
            uncertainty = -np.abs(proba - 0.5)  # higher = more uncertain
            top_k = min(max(10, len(virtuals) // 5), len(virtuals))
            top_idx = np.lexsort((np.arange(len(uncertainty)), -uncertainty))[:top_k]
            filtered = virtuals[top_idx]
        else:
            uncertainty = -np.abs(pred - y_median)
            top_k = min(max(10, len(virtuals) // 5), len(virtuals))
            top_idx = np.lexsort((np.arange(len(uncertainty)), -uncertainty))[:top_k]
            filtered = virtuals[top_idx]
    return np.asarray(filtered)


def _kth_nearest_dists(X_subset: np.ndarray, X_query: np.ndarray, k_max: int) -> np.ndarray:
    """Distances from each query row to its 1st/3rd/5th/10th (``_K_SCALES``) nearest neighbor in ``X_subset``.

    Returns an ``(n_query, len(_K_SCALES))`` matrix. If ``X_subset`` is empty, returns a large sentinel
    distance (1e6) for every scale so downstream log-gap arithmetic stays finite. When ``X_subset`` has
    fewer rows than a requested ``k``, falls back to the farthest available neighbor for that scale.
    """
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


def compute_active_virtual_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    oversample: float = 20.0,
    k_smote: int = 5,
    margin_threshold: float = 0.15,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "actv",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Active virtual placement: SMOTE filtered to aux LGB decision boundary.

    Output: 8 columns per row.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split rows into (positive-cloud, negative-cloud) subsets: class masks for binary, ``q_high``/``1-q_high`` quantile tails for regression."""
        if task == "binary":
            pos = y_sub > 0.5
            return X_sub[pos], X_sub[~pos]
        y_hi = np.quantile(y_sub, q_high)
        y_lo = np.quantile(y_sub, 1.0 - q_high)
        return X_sub[y_sub >= y_hi], X_sub[y_sub <= y_lo]

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        """Run one fold/mode pass: standardize, SMOTE + boundary-filter the positive cloud, then compute per-query k-NN distances and log-gaps.

        Returns an ``(n_query, 2 * len(_K_SCALES))`` matrix: k-NN distances to the boundary-augmented positive
        cloud followed by the signed log-gap vs the negative cloud. Falls back to all-zero features when either
        cloud has fewer than 2 rows (insufficient support to fit the aux LGB or run SMOTE).
        """
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
        # Generate large initial pool — oversample=20x to ensure enough boundary virtuals after filtering.
        n_pool = max(200, int(Xt_pos.shape[0] * oversample))
        raw_virtuals = _smote_synthesize_intra(Xt_pos, n_synthetic=n_pool, k_neighbors=k_smote, seed=fold_seed)
        boundary_virtuals = _filter_boundary_virtuals(Xt_s, y_t, raw_virtuals, task=task, seed=fold_seed, margin_threshold=margin_threshold)
        # Combine real positives + boundary virtuals.
        X_virtual_pos = np.concatenate([Xt_pos, boundary_virtuals], axis=0)
        pos_d = _kth_nearest_dists(X_virtual_pos, Xq_s, max(_K_SCALES))
        neg_d = _kth_nearest_dists(Xt_neg, Xq_s, max(_K_SCALES))
        log_gap = np.log(np.maximum(neg_d, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
        return np.asarray(np.concatenate([pos_d, log_gap], axis=1).astype(np.float32))

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        """Name and cast the ``_process`` output columns to the output ``dtype``, prefixed with ``column_prefix``."""
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
        logger.info("active_virtual: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
