"""Weighted reductions used to build per-head row-attention output features.

These are the post-softmax aggregates. For each query row and head, given the top-k neighbour weights ``w`` (sum to 1 after softmax) and neighbour targets / projected
features, we compute:

- ``y_mean``  - weighted mean of neighbour targets
- ``y_std``   - sqrt of weighted second central moment (biased: divides by sum of weights, not "sum minus one")
- ``x_mean``  - per-dim weighted mean of neighbour projected features (vector aggregate, length head_dim)

The single-row helpers below are kept simple and ``fastmath=False`` — correctness-critical reductions, not the hot path. The fused stage-4 kernel inlines the same
math directly (so it can keep the weights, dots, and aggregates in registers without round-trip loads); these helpers exist for standalone callers (dedupe filter,
biz_value tests, leakage tests) and for parity assertions against the fused kernel.

Decorator rationale: ``fastmath=False`` because the weighted std uses a ``sum(w * (y - mean)^2)`` form that is sensitive to catastrophic cancellation when the
weights are sharply peaked (one neighbour dominates and the rest contribute near-zero) and the mean is large relative to the spread. fastmath would let LLVM
fold the subtractions through FMAs that lose digits in exactly that regime.
"""
from __future__ import annotations

import numba
import numpy as np

NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)


@numba.njit(**NUMBA_NJIT_PARAMS)
def weighted_mean_1d(values: np.ndarray, weights: np.ndarray) -> float:  # pragma: no cover
    """Weighted mean ``sum(w * v) / sum(w)``.

    Returns 0.0 for empty inputs and for all-zero weights (degenerate softmax output where every logit was -inf — should not happen with a numerically stable
    softmax but the guard is cheap and avoids NaN propagation downstream).
    """
    if values.shape[0] == 0:
        return 0.0
    s = 0.0
    w_sum = 0.0
    for i in range(values.shape[0]):
        s += weights[i] * values[i]
        w_sum += weights[i]
    if w_sum <= 0.0:
        return 0.0
    return s / w_sum


@numba.njit(**NUMBA_NJIT_PARAMS)
def weighted_var_1d(values: np.ndarray, weights: np.ndarray) -> float:  # pragma: no cover
    """Weighted (biased) variance ``sum(w * (v - mean)^2) / sum(w)``. Uses a two-pass algorithm so a single sharply-peaked weight doesn't propagate rounding.

    Two-pass over one-pass (``E[v^2] - E[v]^2``) here because the one-pass form has the classic catastrophic-cancellation bug when ``E[v]^2 >> Var(v)`` and the
    neighbour-aggregation regime (one dominant target, k-1 noisy ones at similar magnitude) is exactly that. The 2x compute cost is negligible at k=32-128.
    """
    if values.shape[0] == 0:
        return 0.0
    mean = weighted_mean_1d(values, weights)
    s = 0.0
    w_sum = 0.0
    for i in range(values.shape[0]):
        d = values[i] - mean
        s += weights[i] * d * d
        w_sum += weights[i]
    if w_sum <= 0.0:
        return 0.0
    return s / w_sum


@numba.njit(**NUMBA_NJIT_PARAMS)
def weighted_std_1d(values: np.ndarray, weights: np.ndarray) -> float:  # pragma: no cover
    """Square root of ``weighted_var_1d``. Guards against tiny negative variance from fp rounding before sqrt."""
    var = weighted_var_1d(values, weights)
    if var <= 0.0:
        return 0.0
    return float(np.sqrt(var))


@numba.njit(parallel=True, **NUMBA_NJIT_PARAMS)
def batch_weighted_mean(  # pragma: no cover
    values: np.ndarray,
    weights: np.ndarray,
    out: np.ndarray,
) -> None:
    """Apply ``weighted_mean_1d`` over a batch of query rows.

    Shapes:
        ``values``   - (n_queries, k)         neighbour targets
        ``weights``  - (n_queries, k)         softmax weights, rows sum to ~1
        ``out``      - (n_queries,)           output, pre-allocated

    Parallelised across the query axis. Used by ``_oof._kfold_attention_loop`` and by the biz_value baselines (plain kNN-TE).
    """
    n_queries = values.shape[0]
    for q in numba.prange(n_queries):
        s = 0.0
        w_sum = 0.0
        for i in range(values.shape[1]):
            s += weights[q, i] * values[q, i]
            w_sum += weights[q, i]
        out[q] = s / w_sum if w_sum > 0.0 else 0.0


@numba.njit(parallel=True, **NUMBA_NJIT_PARAMS)
def batch_weighted_std(  # pragma: no cover
    values: np.ndarray,
    weights: np.ndarray,
    out: np.ndarray,
) -> None:
    """Apply ``weighted_std_1d`` over a batch of query rows. Two-pass per query for numerical stability."""
    n_queries = values.shape[0]
    for q in numba.prange(n_queries):
        mean = 0.0
        w_sum = 0.0
        for i in range(values.shape[1]):
            mean += weights[q, i] * values[q, i]
            w_sum += weights[q, i]
        if w_sum <= 0.0:
            out[q] = 0.0
            continue
        mean /= w_sum
        s = 0.0
        for i in range(values.shape[1]):
            d = values[q, i] - mean
            s += weights[q, i] * d * d
        var = s / w_sum
        out[q] = np.sqrt(var) if var > 0.0 else 0.0


@numba.njit(parallel=True, **NUMBA_NJIT_PARAMS)
def batch_weighted_vector_mean(  # pragma: no cover
    values: np.ndarray,
    weights: np.ndarray,
    out: np.ndarray,
) -> None:
    """Per-query weighted mean across a (k, dim) neighbour-feature block.

    Shapes:
        ``values``   - (n_queries, k, dim)    neighbour projected features
        ``weights``  - (n_queries, k)         softmax weights
        ``out``      - (n_queries, dim)       output, pre-allocated

    Used to compute the ``x_mean`` aggregate (per-head projected-feature centroid of neighbours) that gets returned alongside ``y_mean`` / ``y_std``. Streams over
    ``dim`` in the inner loop so the per-query weighted sum is a single sequential pass over neighbour memory rather than k separate passes.
    """
    n_queries, k, dim = values.shape
    for q in numba.prange(n_queries):
        w_sum = 0.0
        for i in range(k):
            w_sum += weights[q, i]
        inv = 1.0 / w_sum if w_sum > 0.0 else 0.0
        for d in range(dim):
            s = 0.0
            for i in range(k):
                s += weights[q, i] * values[q, i, d]
            out[q, d] = s * inv


def compute_extra_aggregates(
    q_proj: np.ndarray,
    k_proj: np.ndarray,
    y_train: np.ndarray,
    topk_ids: np.ndarray,
    softmax_temp: float,
    aggregates: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Compute richer per-query aggregates that the fused stage-4 kernel doesn't emit.

    Implemented at numpy level (not njit) because these are simple per-query reductions and at our k (32-128) the Python loop overhead is negligible. Adding
    them to the fused kernel would balloon the RawKernel source for marginal speedup.

    Supported aggregate names (subset of):
        - ``y_iqr``           - interquartile range of weighted neighbour targets (uncertainty proxy)
        - ``y_skew``          - weighted skewness (asymmetry indicator)
        - ``x_centroid_dist`` - distance from query's projection to the weighted centroid of neighbour projections (cluster density / outlier indicator)

    Returns: dict ``{agg_name: ndarray of shape (n_queries,)}``.

    Why these specifically: CatBoost computes ``y_mean`` analogue internally (ordered target statistics) so our base row-attention output partly overlaps with
    its native features. These three aggregates are NOT in CatBoost's internal feature set and capture different aspects of the kNN neighbourhood structure.
    """
    n_queries, k_count = topk_ids.shape
    head_dim = q_proj.shape[1]

    # Recompute softmax weights from the same logits the fused kernel used (cosine similarity / softmax_temp). This is duplicated work vs the fused kernel but at
    # n_queries * k complexity it's negligible.
    weights = np.empty((n_queries, k_count), dtype=np.float32)
    for q in range(n_queries):
        ids_q = topk_ids[q]
        logits_q = (k_proj[ids_q] @ q_proj[q]) / softmax_temp
        m = float(logits_q.max())
        if not np.isfinite(m):
            weights[q] = 1.0 / k_count
            continue
        exps = np.exp(logits_q - m)
        s = float(exps.sum())
        if s <= 0.0 or not np.isfinite(s):
            weights[q] = 1.0 / k_count
        else:
            weights[q] = exps / s

    out: dict[str, np.ndarray] = {}

    if "y_iqr" in aggregates:
        # Weighted IQR via cumulative-weight quantile. For each query: sort neighbour targets, find values at cumulative weights 0.25 and 0.75.
        y_iqr = np.empty(n_queries, dtype=np.float32)
        for q in range(n_queries):
            y_q = y_train[topk_ids[q]]
            order = np.argsort(y_q)
            y_sorted = y_q[order]
            w_sorted = weights[q][order]
            cum_w = np.cumsum(w_sorted)
            # find 25th and 75th percentile by cumulative weight crossing.
            q25 = float(np.interp(0.25, cum_w, y_sorted))
            q75 = float(np.interp(0.75, cum_w, y_sorted))
            y_iqr[q] = q75 - q25
        out["y_iqr"] = y_iqr

    if "y_skew" in aggregates:
        # Weighted skewness (biased): E[(y - mean)^3] / std^3.
        y_skew = np.empty(n_queries, dtype=np.float32)
        for q in range(n_queries):
            y_q = y_train[topk_ids[q]]
            w_q = weights[q]
            mean = float((w_q * y_q).sum())
            d = y_q - mean
            var = float((w_q * d * d).sum())
            if var <= 1e-12:
                y_skew[q] = 0.0
                continue
            std = np.sqrt(var)
            m3 = float((w_q * d * d * d).sum())
            y_skew[q] = m3 / (std ** 3)
        out["y_skew"] = y_skew

    if "x_centroid_dist" in aggregates:
        # Distance from query projection to weighted centroid of neighbour projections. Larger = query lives further from the cluster centroid =
        # outlier-ish neighbourhood, smaller = query sits inside a dense cluster.
        x_centroid_dist = np.empty(n_queries, dtype=np.float32)
        for q in range(n_queries):
            neighbours = k_proj[topk_ids[q]]  # (k, head_dim)
            centroid = (weights[q][:, None] * neighbours).sum(axis=0)
            diff = q_proj[q] - centroid
            x_centroid_dist[q] = float(np.linalg.norm(diff))
        out["x_centroid_dist"] = x_centroid_dist

    return out


def dedupe_by_correlation(
    features: np.ndarray,
    threshold: float = 0.99,
) -> np.ndarray:
    """Drop one of every pair of columns with ``|corr| > threshold``.

    Used to thin the multi-head-times-multi-aggregate output (typically 50-200 columns, many near-duplicate) before handing to a downstream LightGBM / CatBoost.
    Highly correlated features waste split budget and dilute feature importance without adding predictive signal.

    Strategy: greedy left-to-right; for each column, drop if its absolute Pearson correlation with any previously-kept column exceeds the threshold. Order-dependent
    (the first column in any cluster of near-duplicates is kept) but stable across runs given a fixed input order, and cheap (O(n_cols^2 * n_rows) for the pairwise
    correlations; n_cols ~~ 200 max so the cost is bounded).

    Returns a boolean mask of columns to keep. Caller applies via ``features[:, mask]``.
    """
    if threshold >= 1.0:
        return np.ones(features.shape[1], dtype=bool)
    if threshold <= 0.0:
        # Threshold below zero means "drop everything correlated even weakly" — almost certainly a user mistake; keep only first column.
        mask = np.zeros(features.shape[1], dtype=bool)
        if features.shape[1] > 0:
            mask[0] = True
        return mask
    n_rows, n_cols = features.shape
    if n_cols < 2:
        return np.ones(n_cols, dtype=bool)
    # Standardise once; correlation between standardised columns is just their dot product divided by n_rows.
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    # Guard zero-variance columns: they correlate as NaN with everyone; keep only the first one.
    nonzero_var = stds > 0.0
    if not nonzero_var.any():
        mask = np.zeros(n_cols, dtype=bool)
        mask[0] = True
        return mask
    std_safe = np.where(nonzero_var, stds, 1.0)
    Z = (features - means) / std_safe
    Z[:, ~nonzero_var] = 0.0
    keep = np.ones(n_cols, dtype=bool)
    keep[~nonzero_var] = False
    if nonzero_var.any():
        # Anchor: first non-zero-variance column.
        first_nz = int(np.argmax(nonzero_var))
        keep[:first_nz] = False  # only zero-variance cols before first_nz; already removed
        keep[first_nz] = True
        for j in range(first_nz + 1, n_cols):
            if not keep[j]:
                continue
            # Correlation with already-kept columns to the left.
            kept_left = np.flatnonzero(keep[:j])
            if kept_left.size == 0:
                continue
            corrs = (Z[:, kept_left].T @ Z[:, j]) / n_rows
            if np.any(np.abs(corrs) > threshold):
                keep[j] = False
    return keep
