"""Exact brute-force subset search over SHAP-coalition proxies (numba).

Ports the user's proven incremental-sum kernel (``find_top_n_combinations``) with three corrections:
  1. float64 accumulator (the research had to switch from float32 to avoid running-sum drift over
     ~1e6+ combos -- notebook cell re-ran with ``float_dtype=np.float64``).
  2. per-row base value (OOF folds carry distinct base values) instead of a scalar.
  3. a *proper* objective (``score_margin``: Brier/log-loss/RMSE/MAE) applied per combo instead of
     MAE-of-0/1-labels-vs-log-odds-margin.

The incremental trick: combinations of a fixed cardinality ``r`` share long prefixes; ``level_sums``
caches the running selected-phi sum per prefix position, so each new combo only re-adds the suffix
past the first index that changed. Correctness is order-independent (we recompute from the first
differing index every time); the recursive generator just makes shared prefixes frequent.

Two kernels kept (keep-all-kernels rule): serial ``_topn_fixed_r`` and chunk-parallel
``_topn_fixed_r_parallel`` (prange over independent combo chunks, each with private accumulators).
Exact enumeration is intended for ``n_features <= ~22``; larger n routes to the heuristics module.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import METRIC_CODES, resolve_metric, score_margin


@njit(cache=True)
def generate_combinations(sequence: np.ndarray, r: int) -> np.ndarray:
    """All length-``r`` combinations of ``sequence`` as a (C, r) int array (recursive, njit)."""
    if r == 0:
        return np.empty((1, 0), dtype=sequence.dtype)
    if sequence.size == 0:
        return np.empty((0, r), dtype=sequence.dtype)
    first, rest = sequence[0], sequence[1:]
    without_first = generate_combinations(rest, r)
    with_first = generate_combinations(rest, r - 1)
    result = np.empty((without_first.shape[0] + with_first.shape[0], r), dtype=sequence.dtype)
    result[: without_first.shape[0], :] = without_first
    for i in range(with_first.shape[0]):
        result[i + without_first.shape[0], 0] = first
        result[i + without_first.shape[0], 1:] = with_first[i, :]
    return result


@njit(cache=True)
def _insert_top(top_losses: np.ndarray, top_combos: np.ndarray, loss: float, comb: np.ndarray) -> None:
    """Insert (loss, comb) into an ascending-by-loss top-N buffer; drop the current worst."""
    top_n = top_losses.shape[0]
    r = comb.shape[0]
    if loss >= top_losses[top_n - 1]:
        return
    top_losses[top_n - 1] = loss
    for k in range(r):
        top_combos[top_n - 1, k] = comb[k]
    for k in range(top_n - 1, 0, -1):
        if top_losses[k] < top_losses[k - 1]:
            tl = top_losses[k]; top_losses[k] = top_losses[k - 1]; top_losses[k - 1] = tl
            for c in range(r):
                tc = top_combos[k, c]; top_combos[k, c] = top_combos[k - 1, c]; top_combos[k - 1, c] = tc
        else:
            break


@njit(cache=True)
def _topn_fixed_r(phi, base, y, combos, metric_code, top_n):
    """Serial top-N scan over fixed-cardinality combinations with incremental prefix sums.

    Row-major variant: ``phi`` shape ``(n, f)``. Kept for backward compatibility and rollback
    benchmarking; ``brute_force_top_n`` routes to the column-major variant by default since
    ``phi[t, fcol]`` strided access here is 2-3x slower than the contiguous column-major form.
    """
    C = combos.shape[0]
    r = combos.shape[1]
    n = phi.shape[0]
    top_n = min(top_n, C)
    top_combos = -np.ones((top_n, r), dtype=np.int32)
    top_losses = np.full(top_n, np.inf, dtype=np.float64)
    prev = np.full(r, -1, dtype=np.int32)
    main_sum = np.zeros(n, dtype=np.float64)
    level_sums = np.zeros((r, n), dtype=np.float64)
    margin = np.empty(n, dtype=np.float64)
    for c in range(C):
        comb = combos[c]
        for i in range(r - 1):
            if comb[i] != prev[i]:
                if i > 0:
                    main_sum[:] = level_sums[i - 1]
                else:
                    main_sum[:] = 0.0
                for j in range(i, r - 1):
                    fcol = comb[j]
                    for t in range(n):
                        main_sum[t] += phi[t, fcol]
                    level_sums[j][:] = main_sum
                break
        last = comb[r - 1]
        for t in range(n):
            margin[t] = base[t] + main_sum[t] + phi[t, last]
        loss = score_margin(margin, y, metric_code)
        _insert_top(top_losses, top_combos, loss, comb)
        prev[:] = comb
    return top_combos, top_losses


@njit(cache=True)
def _topn_fixed_r_colmajor(phi_T, base, y, combos, metric_code, top_n):
    """Column-major variant of ``_topn_fixed_r``: ``phi_T`` shape ``(f, n)`` so each feature column is
    a contiguous row, making the inner ``phi_T[fcol]`` access a unit-stride read instead of the
    strided ``phi[:, fcol]`` of the row-major variant. End-to-end speedup on f=18, n=1000, r=6:
    ~2.5-3x for MAE/MSE (memory-bound), wash for Brier/log-loss (exp-bound) -- selected as default.
    """
    C = combos.shape[0]
    r = combos.shape[1]
    n = phi_T.shape[1]
    top_n = min(top_n, C)
    top_combos = -np.ones((top_n, r), dtype=np.int32)
    top_losses = np.full(top_n, np.inf, dtype=np.float64)
    prev = np.full(r, -1, dtype=np.int32)
    main_sum = np.zeros(n, dtype=np.float64)
    level_sums = np.zeros((r, n), dtype=np.float64)
    margin = np.empty(n, dtype=np.float64)
    for c in range(C):
        comb = combos[c]
        for i in range(r - 1):
            if comb[i] != prev[i]:
                if i > 0:
                    main_sum[:] = level_sums[i - 1]
                else:
                    main_sum[:] = 0.0
                for j in range(i, r - 1):
                    fcol_row = phi_T[comb[j]]
                    for t in range(n):
                        main_sum[t] += fcol_row[t]
                    level_sums[j][:] = main_sum
                break
        last_row = phi_T[comb[r - 1]]
        for t in range(n):
            margin[t] = base[t] + main_sum[t] + last_row[t]
        loss = score_margin(margin, y, metric_code)
        _insert_top(top_losses, top_combos, loss, comb)
        prev[:] = comb
    return top_combos, top_losses


@njit(cache=True, parallel=True)
def _topn_fixed_r_parallel(phi, base, y, combos, metric_code, top_n, n_chunks):
    """Chunk-parallel variant: each prange chunk runs an independent incremental scan with private
    accumulators, producing a local top-N; locals are concatenated for a serial final merge.

    Row-major variant (``phi`` shape ``(n, f)``). Kept for backward compatibility; the dispatcher in
    ``brute_force_top_n`` routes parallel paths to ``_topn_fixed_r_parallel_colmajor`` for the same
    cache-locality reason as the serial kernel.
    """
    C = combos.shape[0]
    r = combos.shape[1]
    n = phi.shape[0]
    top_n = min(top_n, C)
    chunk_losses = np.full((n_chunks, top_n), np.inf, dtype=np.float64)
    chunk_combos = -np.ones((n_chunks, top_n, r), dtype=np.int32)
    base_chunk = C // n_chunks
    for ch in prange(n_chunks):
        start = ch * base_chunk
        stop = C if ch == n_chunks - 1 else (ch + 1) * base_chunk
        prev = np.full(r, -1, dtype=np.int32)
        main_sum = np.zeros(n, dtype=np.float64)
        level_sums = np.zeros((r, n), dtype=np.float64)
        margin = np.empty(n, dtype=np.float64)
        local_losses = chunk_losses[ch]
        local_combos = chunk_combos[ch]
        for c in range(start, stop):
            comb = combos[c]
            for i in range(r - 1):
                if comb[i] != prev[i]:
                    if i > 0:
                        main_sum[:] = level_sums[i - 1]
                    else:
                        main_sum[:] = 0.0
                    for j in range(i, r - 1):
                        fcol = comb[j]
                        for t in range(n):
                            main_sum[t] += phi[t, fcol]
                        level_sums[j][:] = main_sum
                    break
            last = comb[r - 1]
            for t in range(n):
                margin[t] = base[t] + main_sum[t] + phi[t, last]
            loss = score_margin(margin, y, metric_code)
            _insert_top(local_losses, local_combos, loss, comb)
            prev[:] = comb
    return chunk_combos, chunk_losses


@njit(cache=True, parallel=True)
def _topn_fixed_r_parallel_colmajor(phi_T, base, y, combos, metric_code, top_n, n_chunks):
    """Column-major chunk-parallel variant: ``phi_T`` shape ``(f, n)``. Same per-thread structure as
    ``_topn_fixed_r_parallel`` but with contiguous-column reads (see ``_topn_fixed_r_colmajor``)."""
    C = combos.shape[0]
    r = combos.shape[1]
    n = phi_T.shape[1]
    top_n = min(top_n, C)
    chunk_losses = np.full((n_chunks, top_n), np.inf, dtype=np.float64)
    chunk_combos = -np.ones((n_chunks, top_n, r), dtype=np.int32)
    base_chunk = C // n_chunks
    for ch in prange(n_chunks):
        start = ch * base_chunk
        stop = C if ch == n_chunks - 1 else (ch + 1) * base_chunk
        prev = np.full(r, -1, dtype=np.int32)
        main_sum = np.zeros(n, dtype=np.float64)
        level_sums = np.zeros((r, n), dtype=np.float64)
        margin = np.empty(n, dtype=np.float64)
        local_losses = chunk_losses[ch]
        local_combos = chunk_combos[ch]
        for c in range(start, stop):
            comb = combos[c]
            for i in range(r - 1):
                if comb[i] != prev[i]:
                    if i > 0:
                        main_sum[:] = level_sums[i - 1]
                    else:
                        main_sum[:] = 0.0
                    for j in range(i, r - 1):
                        fcol_row = phi_T[comb[j]]
                        for t in range(n):
                            main_sum[t] += fcol_row[t]
                        level_sums[j][:] = main_sum
                    break
            last_row = phi_T[comb[r - 1]]
            for t in range(n):
                margin[t] = base[t] + main_sum[t] + last_row[t]
            loss = score_margin(margin, y, metric_code)
            _insert_top(local_losses, local_combos, loss, comb)
            prev[:] = comb
    return chunk_combos, chunk_losses


def _merge_topn(
    candidates: list[tuple[float, tuple[int, ...]]],
    top_n: int,
    keep_best_per_card: bool = True,
) -> list[tuple[float, tuple[int, ...]]]:
    """Dedup candidates, keep the global top-N, and (default) force-include the single best subset of
    every cardinality present. Per-cardinality coverage matters because the proxy loss is monotonically
    non-increasing as features are added (more phi to fit y), so a pure global top-N skews toward large
    subsets and starves the parsimonious sizes that honest re-validation needs to compare against."""
    seen: dict[tuple[int, ...], float] = {}
    for loss, comb in candidates:
        if not np.isfinite(loss):
            continue
        key = tuple(sorted(comb))
        if key not in seen or loss < seen[key]:
            seen[key] = loss
    merged = sorted(((loss, comb) for comb, loss in seen.items()), key=lambda t: t[0])
    kept = merged[:top_n]
    if keep_best_per_card:
        per_card: dict[int, tuple[float, tuple[int, ...]]] = {}
        for loss, comb in merged:
            r = len(comb)
            if r not in per_card or loss < per_card[r][0]:
                per_card[r] = (loss, comb)
        kept_keys = {comb for _, comb in kept}
        for loss, comb in per_card.values():
            if comb not in kept_keys:
                kept.append((loss, comb))
        kept.sort(key=lambda t: t[0])
    return kept


def brute_force_top_n(
    phi: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    metric: str | None = None,
    min_card: int = 1,
    max_card: int | None = None,
    top_n: int = 30,
    parallel: bool = False,
    n_chunks: int = 8,
) -> list[tuple[float, tuple[int, ...]]]:
    """Exhaustively rank feature subsets by proxy loss; return the top-N as (loss, feature_idx tuple).

    Enumerates every cardinality in ``[min_card, max_card]`` and merges per-cardinality winners.
    """
    metric_name = resolve_metric(classification, metric)
    if metric_name == "auc":  # AUC needs a per-subset sort -> not in the njit hot loop
        raise ValueError("brute_force_top_n does not support metric='auc' (use brier/logloss); AUC is Python-path only.")
    metric_code = METRIC_CODES[metric_name]
    phi = np.ascontiguousarray(phi, dtype=np.float64)
    base = np.ascontiguousarray(base, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_features = phi.shape[1]
    max_card = n_features if max_card is None else min(max_card, n_features)
    # Transpose once: column-major (f, n) layout turns the strided ``phi[t, fcol]`` reads inside the
    # numba hot loop into unit-stride row reads, ~2-3x faster end-to-end at f=18, n=1000, r=6 for
    # MAE/MSE (memory-bound) and neutral for Brier/log-loss (exp-bound).
    phi_T = np.ascontiguousarray(phi.T)

    seq = np.arange(n_features, dtype=np.int32)
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for r in range(min_card, max_card + 1):
        if math.comb(n_features, r) == 0:
            continue
        combos = generate_combinations(seq, r)
        if combos.shape[0] == 0:
            continue
        if parallel and combos.shape[0] >= n_chunks * 4:
            ch_combos, ch_losses = _topn_fixed_r_parallel_colmajor(phi_T, base, y, combos, metric_code, top_n, n_chunks)
            for ch in range(ch_combos.shape[0]):
                for k in range(ch_combos.shape[1]):
                    if np.isfinite(ch_losses[ch, k]):
                        candidates.append((float(ch_losses[ch, k]), tuple(int(x) for x in ch_combos[ch, k])))
        else:
            tc, tl = _topn_fixed_r_colmajor(phi_T, base, y, combos, metric_code, top_n)
            for k in range(tc.shape[0]):
                if np.isfinite(tl[k]):
                    candidates.append((float(tl[k]), tuple(int(x) for x in tc[k])))
    merged = _merge_topn(candidates, top_n)
    if metric_name == "rmse":  # kernel computes MSE (rank-equivalent); report the sqrt to match the name
        merged = [(float(np.sqrt(loss)), comb) for loss, comb in merged]
    return merged


def total_subsets(n_features: int, min_card: int, max_card: int | None) -> int:
    """Number of subsets an exhaustive run would evaluate (for the dispatcher's feasibility check)."""
    max_card = n_features if max_card is None else min(max_card, n_features)
    return int(sum(math.comb(n_features, r) for r in range(min_card, max_card + 1)))
