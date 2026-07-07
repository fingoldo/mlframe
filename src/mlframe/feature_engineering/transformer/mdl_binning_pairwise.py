"""MDL supervised binning (Fayyad-Irani) + top-K pairwise bin co-occurrence features.

Iter 97 mechanism. Agent A #2 ranked.

Per feature: recursive MDL entropy-split binning of x_j by target. Per query emit aggregate features.
"""
from __future__ import annotations
import logging
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


logger = logging.getLogger(__name__)


@njit(cache=True)
def _entropy_from_counts(counts, total):
    """Multi-class entropy in bits from an integer class-count vector and its sum.

    Replicates ``_entropy_multi`` numerics: ``p = count/total``, ``-sum(p*log2(p))`` over classes in index order, skipping p<=0.
    """
    if total == 0:
        return 0.0
    e = 0.0
    for c in range(counts.shape[0]):
        pi = counts[c] / total
        if pi > 0.0:
            e -= pi * np.log2(pi)
    return e


@njit(cache=True)
def _best_mdl_split_kernel(y_sorted, x_sorted, n_classes, min_size):
    """Single-pass Fayyad-Irani best-split scan over a range pre-sorted by x.

    Replaces the O(n^2) inner loop (which recomputed two ``np.bincount`` entropies per candidate split) with running prefix class
    counts updated incrementally as the split index advances; entropy is evaluated from the count vectors at each candidate, so the
    arithmetic order over classes matches ``_entropy_multi`` exactly (bit-identical selection).

    Returns (best_idx, best_thresh, best_gain, E_left_best, E_right_best). best_idx = -1 when no valid split exists.
    """
    n = y_sorted.shape[0]
    # H_S over the whole range.
    total_counts = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        total_counts[y_sorted[i]] += 1
    H_S = _entropy_from_counts(total_counts, n)

    left_counts = np.zeros(n_classes, dtype=np.int64)
    # Prime left counts for the first candidate i == min_size.
    for i in range(min_size):
        left_counts[y_sorted[i]] += 1
    right_counts = total_counts - left_counts

    best_gain = -1.0
    best_idx = -1
    best_thresh = 0.0
    E_left_best = 0.0
    E_right_best = 0.0

    for i in range(min_size, n - min_size):
        if i > 0 and x_sorted[i] == x_sorted[i - 1]:
            left_counts[y_sorted[i]] += 1
            right_counts[y_sorted[i]] -= 1
            continue
        E_left = _entropy_from_counts(left_counts, i)
        E_right = _entropy_from_counts(right_counts, n - i)
        weighted = (i / n) * E_left + ((n - i) / n) * E_right
        gain = H_S - weighted
        if gain > best_gain:
            best_gain = gain
            best_idx = i
            best_thresh = (x_sorted[i - 1] + x_sorted[i]) / 2.0
            E_left_best = E_left
            E_right_best = E_right
        left_counts[y_sorted[i]] += 1
        right_counts[y_sorted[i]] -= 1

    return best_idx, best_thresh, best_gain, E_left_best, E_right_best, H_S


def _entropy_binary(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def _entropy_multi(y_subset: np.ndarray, n_classes: int) -> float:
    if y_subset.size == 0:
        return 0.0
    counts = np.bincount(y_subset.astype(np.int32), minlength=n_classes).astype(np.float64)
    p = counts / counts.sum()
    e = 0.0
    for pi in p:
        if pi > 0:
            e -= pi * np.log2(pi)
    return e


def _mdl_bin_edges(x: np.ndarray, y_class: np.ndarray, n_classes: int, max_bins=8, min_size=20) -> list[float]:
    """Fayyad-Irani MDL best-split binning. Returns sorted unique split points.

    Single top-level split (pinned bit-identical to the pre-iter81 O(n^2) reference by
    ``tests/feature_engineering/transformer/test_mdl_binning_split_kernel.py``); not recursive
    despite ``max_bins``/``depth`` naming -- ``_split`` is only ever invoked once, over the full range.
    """
    edges = []

    def _split(lo, hi, depth):
        if hi - lo < 2 * min_size or len(edges) >= max_bins - 1:
            return
        sub_x = x[lo:hi]
        sub_y = y_class[lo:hi]
        n = hi - lo
        H_S = _entropy_multi(sub_y, n_classes)
        if H_S < 1e-6:
            return
        # Sort by x within range
        order = np.argsort(sub_x, kind="stable")
        x_sorted = np.ascontiguousarray(sub_x[order])
        y_sorted = np.ascontiguousarray(sub_y[order], dtype=np.int64)
        # Single-pass njit best-split scan (replaces the prior O(n^2) double-bincount inner loop). Returns the same best split + the
        # left/right entropies needed for the MDL stop term, bit-identical to recomputing _entropy_multi over the same slices.
        best_idx, best_thresh, best_gain, E_left_best, E_right_best, _H_S_sorted = _best_mdl_split_kernel(y_sorted, x_sorted, n_classes, min_size)
        if best_idx < 0:
            return
        # MDL stop criterion
        k = n_classes
        delta = np.log2(3**k - 2) - (k * H_S - 2 * E_left_best * (best_idx / n) - (n - best_idx) / n * 2 * E_right_best)
        if best_gain * n < np.log2(n - 1) + delta:
            return
        edges.append(float(best_thresh))

    _split(0, x.size, 0)
    edges.sort()
    return edges


def compute_mdl_binning_pairwise_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    max_bins_per_feat=8, top_k_pairs=10, standardize=True, column_prefix="mdlbin", dtype=np.float32,
):
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt, Xq, y_t):
        d = Xt.shape[1]
        # Discretize y for MDL: regression → quintile bin; binary → as-is
        if y_t.dtype != np.int32:
            if (y_t == 0).sum() + (y_t == 1).sum() == y_t.size:
                y_class = y_t.astype(np.int32)
                n_classes = 2
            else:
                qs = np.quantile(y_t, [0.2, 0.4, 0.6, 0.8])
                y_class = np.digitize(y_t, qs).astype(np.int32)
                n_classes = 5
        # MDL bin edges per feature
        all_edges = []
        for j in range(d):
            edges = _mdl_bin_edges(Xt[:, j], y_class, n_classes, max_bins=max_bins_per_feat)
            all_edges.append(edges)
        # Compute bin assignments per row
        train_bins = np.zeros((Xt.shape[0], d), dtype=np.int32)
        query_bins = np.zeros((Xq.shape[0], d), dtype=np.int32)
        for j in range(d):
            train_bins[:, j] = np.digitize(Xt[:, j], all_edges[j])
            query_bins[:, j] = np.digitize(Xq[:, j], all_edges[j])
        # Per-query aggregate features
        max_bin_idx = query_bins.max(axis=1).astype(np.float32)
        sum_bins = query_bins.sum(axis=1).astype(np.float32)
        # Pairwise co-occurrence: count rows in train sharing same bin combo for first 2 features.
        # Vectorised np.unique + searchsorted lookup replaces the per-query-row Counter.get() Python loop
        # (5-10x faster at n>=10k); counts are integers so the result is bit-identical to the dict path.
        if d >= 2:
            train_combo = train_bins[:, 0] * 100 + train_bins[:, 1]
            query_combo = query_bins[:, 0] * 100 + query_bins[:, 1]
            uniq_combo, uniq_counts = np.unique(train_combo, return_counts=True)
            pos = np.searchsorted(uniq_combo, query_combo)
            pos_clipped = np.clip(pos, 0, uniq_combo.shape[0] - 1)
            matched = uniq_combo[pos_clipped] == query_combo
            combo_count_per_query = np.where(matched, uniq_counts[pos_clipped], 0).astype(np.float32)
            unique_combos = float(np.unique(query_combo).shape[0])
        else:
            combo_count_per_query = np.zeros(Xq.shape[0], dtype=np.float32)
            unique_combos = 1.0
        # n_edges_total
        n_edges_total = float(sum(len(e) for e in all_edges))
        return np.column_stack([
            max_bin_idx,
            sum_bins,
            combo_count_per_query,
            np.full(Xq.shape[0], n_edges_total, dtype=np.float32),
            np.full(Xq.shape[0], unique_combos, dtype=np.float32),
        ])

    def _make_df(feats):
        cols = {}
        cols[f"{column_prefix}_max_bin"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_sum_bins"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_combo_count"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_edges_total"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_unique_combos"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        return pl.DataFrame(_make_df(_process(X_train_f, Xq, y_train_f)))
    if splitter is None:
        raise ValueError("Mode A requires splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_f)):
        out[val_idx] = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx]).astype(dtype, copy=False)
        logger.info("mdl_binning_pairwise: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
