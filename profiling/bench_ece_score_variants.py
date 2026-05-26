"""Bench ECE score variants: numpy bincount vs numba serial vs numba parallel.

iter309 (2026-05-26) follow-up to iter308: comparing alternative
backends for ``_ece_score`` to pick the actually-fastest default.
Run: ``python profiling/bench_ece_score_variants.py``.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange


def _ece_numpy(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Current shipped (663edd53) numpy bincount path."""
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(p) & np.isfinite(y)
    if not finite.any():
        return float("nan")
    p = p[finite]
    y = y[finite]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] = np.nextafter(1.0, 2.0)
    bin_ids = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)
    sum_p = np.bincount(bin_ids, weights=p, minlength=n_bins)
    sum_y = np.bincount(bin_ids, weights=y, minlength=n_bins)
    return float(np.abs(sum_y - sum_p).sum() / p.size)


@njit(cache=True, nogil=True)
def _ece_numba(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    n = p.size
    counts = np.zeros(n_bins, dtype=np.float64)
    sum_p = np.zeros(n_bins, dtype=np.float64)
    sum_y = np.zeros(n_bins, dtype=np.float64)
    for i in range(n):
        pi = p[i]
        yi = y[i]
        if not (np.isfinite(pi) and np.isfinite(yi)):
            continue
        b = int(pi * n_bins)
        if b >= n_bins:
            b = n_bins - 1
        elif b < 0:
            b = 0
        counts[b] += 1.0
        sum_p[b] += pi
        sum_y[b] += yi
    n_finite = 0.0
    for b in range(n_bins):
        n_finite += counts[b]
    if n_finite == 0:
        return float("nan")
    total = 0.0
    for b in range(n_bins):
        diff = sum_y[b] - sum_p[b]
        if diff < 0.0:
            diff = -diff
        total += diff
    return total / n_finite


@njit(cache=True, parallel=True, nogil=True)
def _ece_numba_par(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Parallel reduction via per-thread bins, then sum."""
    n = p.size
    # numba.prange-friendly: allocate per-bin counts; use manual reduction.
    # Numba's reduction operator on numpy arrays inside prange is buggy across
    # versions, so we materialise per-iter bin assignment then sequentially
    # reduce. The forward scan (bin_id computation) is the dominant cost on
    # large n and is the only part that is genuinely parallel.
    bin_ids = np.empty(n, dtype=np.int64)
    finite = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        pi = p[i]
        yi = y[i]
        if np.isfinite(pi) and np.isfinite(yi):
            b = int(pi * n_bins)
            if b >= n_bins:
                b = n_bins - 1
            elif b < 0:
                b = 0
            bin_ids[i] = b
            finite[i] = True
        else:
            bin_ids[i] = -1
            finite[i] = False
    counts = np.zeros(n_bins, dtype=np.float64)
    sum_p = np.zeros(n_bins, dtype=np.float64)
    sum_y = np.zeros(n_bins, dtype=np.float64)
    n_finite = 0.0
    for i in range(n):
        if finite[i]:
            b = bin_ids[i]
            counts[b] += 1.0
            sum_p[b] += p[i]
            sum_y[b] += y[i]
            n_finite += 1.0
    if n_finite == 0.0:
        return float("nan")
    total = 0.0
    for b in range(n_bins):
        diff = sum_y[b] - sum_p[b]
        if diff < 0.0:
            diff = -diff
        total += diff
    return total / n_finite


def _bench(n: int, N: int, label: str) -> None:
    print(f"\n--- {label} (n={n}) ---")
    np.random.seed(0)
    y = np.random.randint(0, 2, n).astype(np.float64)
    p = np.random.rand(n).astype(np.float64)
    # Warmup
    r_np = _ece_numpy(y, p)
    r_nb = _ece_numba(y, p)
    r_pa = _ece_numba_par(y, p)
    assert abs(r_np - r_nb) < 1e-12, f"numba mismatch: {r_np} vs {r_nb}"
    assert abs(r_np - r_pa) < 1e-12, f"parallel mismatch: {r_np} vs {r_pa}"
    t0 = time.perf_counter()
    for _ in range(N):
        _ece_numpy(y, p)
    t1 = time.perf_counter()
    for _ in range(N):
        _ece_numba(y, p)
    t2 = time.perf_counter()
    for _ in range(N):
        _ece_numba_par(y, p)
    t3 = time.perf_counter()
    print(f"  numpy bincount: {(t1 - t0) * 1000 / N:7.4f} ms/call")
    print(f"  numba serial:   {(t2 - t1) * 1000 / N:7.4f} ms/call  ({(t1 - t0) / (t2 - t1):4.2f}x vs numpy)")
    print(f"  numba parallel: {(t3 - t2) * 1000 / N:7.4f} ms/call  ({(t1 - t0) / (t3 - t2):4.2f}x vs numpy)")


def main() -> None:
    _bench(2_000, 500, "small (typical bootstrap resample)")
    _bench(20_000, 200, "medium (full bootstrap sample)")
    _bench(200_000, 30, "large (single-call full split)")
    _bench(1_000_000, 10, "x-large (production scale)")


if __name__ == "__main__":
    main()
