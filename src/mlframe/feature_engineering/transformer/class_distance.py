"""Class-distance / quantile-distance attention: per-row distances to nearest k-th class-specific instances.

Iter 27 mechanism. Targets the CB AUC mammography ceiling and abalone all-3-positive magnitudes.

CatBoost's TS encoding computes ``E[y | feature_j]`` per column; CB's symmetric oblivious trees split on per-feature thresholds; CB has NO way to compute "distance from this
row to the nearest known positive-class instance". That geometric quantity requires per-instance kNN lookup — fundamentally different from per-feature aggregation.

Mechanism (binary):
- For each query row, find the k-th nearest POSITIVE-class training row and the k-th nearest NEGATIVE-class training row, at multiple k ∈ {1, 3, 5, 10}.
- Expose 8 features: 4 positive-class distances + 4 negative-class distances.
- Plus a signed log-gap: ``log(d_neg / d_pos)`` per k-scale — the classic Bayes-decision-rule-aligned ratio. 4 more features.
- Total: 12 features per row for binary.

Mechanism (regression):
- Define "high-y" = top quintile of y_train (q=0.8); "low-y" = bottom quintile (q=0.2).
- Distances to nearest k=1,3,5,10-th high-y row, plus nearest k=1,3,5,10-th low-y row, plus signed log-gap.
- Total: 12 features per row for regression.

The signed log-gap is a Bayes-rule-aligned feature: under equal class densities, the optimal classifier is sign(log d_neg - log d_pos) = sign of log-density ratio approximation.
Boostings can ALSO derive this approximately from the raw distances, but having it pre-computed as one column removes the burden from their split budget.

Leakage discipline:
- Mode A (X_query=None): per-fold refit. Class-sliced training sub-arrays computed from y_train[train_idx] only. Val rows queried against fold's class-sliced banks.
- Mode B (X_query given): class-sliced training sub-arrays from full y_train; query rows scored.

Cost: 2 kNN searches per fold (one against positives, one against negatives), each O(N_class · d). For mammography (52 positives, 3948 negatives), positive-kNN is essentially free.

Why this differs from iter 19 (class-conditional anchor):
- Iter 19 uses K-MEANS CENTROIDS per class — averaged anchor representation.
- Iter 27 uses RAW NEAREST INSTANCES — exposes outliers and rare-cluster members the K-means averaging hides.

For mammography (1.3% positive, very rare): K-means on positives gives ~16 centroids that lose individual-row geometry. Raw nearest-instance distance preserves it.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._intel_patch import try_patch_sklearn
from ._knn_helper import knn_search
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _kth_nearest_dists(X_subset: np.ndarray, X_query: np.ndarray, k_max: int) -> np.ndarray:
    """Return per-query distances to the 1st, 3rd, 5th, 10th nearest rows of X_subset (or fewer if X_subset too small).

    Shape: (n_query, len(_K_SCALES)). Missing scales (when X_subset smaller than scale) filled with the max-available-k distance.

    Uses ``_knn_helper.knn_search`` which auto-dispatches to hnswlib at N>=50000 (10-50x speedup
    on Windows when hnswlib is installed) and falls back to sklearn NearestNeighbors otherwise.
    """
    n_sub = X_subset.shape[0]
    if n_sub == 0:
        return np.full((X_query.shape[0], len(_K_SCALES)), 1e6, dtype=np.float32)
    dists, _ids = knn_search(X_subset, X_query, k=k_max)
    n_returned = dists.shape[1]
    out = np.zeros((X_query.shape[0], len(_K_SCALES)), dtype=np.float32)
    for col_idx, k in enumerate(_K_SCALES):
        eff_k = min(k, n_returned)
        out[:, col_idx] = dists[:, eff_k - 1]
    return out


def _compute_features(
    X_train_subset_pos: np.ndarray,
    X_train_subset_neg: np.ndarray,
    X_query: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (pos_dists, neg_dists, log_gap) features for query rows.

    pos_dists: (n_q, |_K_SCALES|) — distances to k-th nearest positive-class row.
    neg_dists: (n_q, |_K_SCALES|) — distances to k-th nearest negative-class row.
    log_gap: (n_q, |_K_SCALES|) — log(d_neg / d_pos) per k-scale.
    """
    k_max = max(_K_SCALES)
    pos_d = _kth_nearest_dists(X_train_subset_pos, X_query, k_max)
    neg_d = _kth_nearest_dists(X_train_subset_neg, X_query, k_max)
    # log-gap with small epsilon to avoid log(0).
    log_gap = np.log(np.maximum(neg_d, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
    return pos_d, neg_d, log_gap.astype(np.float32)


def _binary_class_slices(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Slice X by binary y. Returns (X_positives, X_negatives)."""
    pos_mask = y > 0.5
    return X[pos_mask], X[~pos_mask]


def _quantile_slices(X: np.ndarray, y: np.ndarray, q_low: float = 0.2, q_high: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Slice X by regression-y quantile bounds. Returns (X_high_quantile, X_low_quantile)."""
    y_lo = np.quantile(y, q_low)
    y_hi = np.quantile(y, q_high)
    return X[y >= y_hi], X[y <= y_lo]


def compute_class_distance_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    standardize: bool = True,
    q_low: float = 0.2,
    q_high: float = 0.8,
    column_prefix: str = "cdist",
    dtype: np.dtype = np.float32,
    exclude_self_ids: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    """Class-distance / quantile-distance attention features.

    Output: 12 columns per row (4 pos-k-distances + 4 neg-k-distances + 4 log-gaps), where "pos" / "neg" are class labels for binary
    or top/bottom y-quantile for regression.

    Mode A: per-class slices refit per fold from y_train[train_idx]. Mode B: slices from full y_train.

    Self-label leakage (Mode B only): when the query rows are a SUBSET of (or overlap) the training rows, a query row finds ITSELF as its own nearest class-mate (distance 0), leaking its own label. Pass ``exclude_self_ids`` -- a 1-D boolean mask of length ``len(X_train)`` that is True for the training rows ALSO present as query rows -- to drop those rows from the class banks before the kNN search. Mode B's documented precondition is a disjoint query set; ``exclude_self_ids`` is the explicit opt-out when that cannot be guaranteed. (Mode A is already leakage-safe via per-fold class slices.)
    """
    seed = require_seed(seed)
    try_patch_sklearn()
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression'; got {task!r}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if task == "binary":
            return _binary_class_slices(X_sub, y_sub)
        return _quantile_slices(X_sub, y_sub, q_low=q_low, q_high=q_high)

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            n_q = Xq_s.shape[0]
            zero = np.zeros((n_q, len(_K_SCALES)), dtype=np.float32)
            return zero, zero, zero
        return _compute_features(Xt_pos, Xt_neg, Xq_s)

    def _make_df(pos_d: np.ndarray, neg_d: np.ndarray, log_gap: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_pos_k{k}"] = pos_d[:, j].astype(dtype, copy=False)
            cols[f"{column_prefix}_neg_k{k}"] = neg_d[:, j].astype(dtype, copy=False)
            cols[f"{column_prefix}_loggap_k{k}"] = log_gap[:, j].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        Xt_bank = X_train_f
        yt_bank = y_train_f
        if exclude_self_ids is not None:
            _mask = np.asarray(exclude_self_ids).ravel()
            if _mask.shape[0] != X_train_f.shape[0]:
                raise ValueError(f"exclude_self_ids length {_mask.shape[0]} != len(X_train) {X_train_f.shape[0]}.")
            _keep = ~_mask.astype(bool)
            Xt_bank = X_train_f[_keep]
            yt_bank = y_train_f[_keep]
        pos_d, neg_d, log_gap = _process(Xt_bank, Xq, yt_bank)
        return pl.DataFrame(_make_df(pos_d, neg_d, log_gap))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    if exclude_self_ids is not None:
        raise ValueError("exclude_self_ids is only meaningful in Mode B (X_query given); Mode A is leakage-safe via per-fold class slices.")
    n_train = X_train_f.shape[0]
    out_pos = np.zeros((n_train, len(_K_SCALES)), dtype=dtype)
    out_neg = np.zeros((n_train, len(_K_SCALES)), dtype=dtype)
    out_gap = np.zeros((n_train, len(_K_SCALES)), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        pos_d, neg_d, log_gap = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out_pos[val_idx] = pos_d
        out_neg[val_idx] = neg_d
        out_gap[val_idx] = log_gap
        logger.info("class_distance: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out_pos, out_neg, out_gap))
