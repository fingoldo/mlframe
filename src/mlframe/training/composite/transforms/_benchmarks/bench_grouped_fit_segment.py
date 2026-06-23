"""Bench: per-group OLS loop in ``_linear_residual_grouped_fit`` -- O(K*n) boolean
masking + fancy-index gathers (OLD) vs a single stable-argsort + contiguous
segment slicing (NEW).

OLD: for each of K unique groups, ``g_mask = (inverse_idx == i)`` is a full-length
boolean scan and ``y[g_mask]`` / ``base[g_mask]`` are fancy-index gathers -> the
whole pre-fit bookkeeping is O(K*n). NEW: sort the rows once by group label
(``np.argsort(inverse_idx, kind="stable")``), derive per-group contiguous
``[start:stop]`` slices from ``np.bincount`` offsets, and view each segment.

Stable argsort preserves the original ascending row order WITHIN each group, so
``y_g`` / ``base_g`` are the SAME arrays (same values, same order) the mask path
produced -> the per-group ``_linear_residual_fit`` (lstsq) result is BIT-IDENTICAL.
Run::

    CUDA_VISIBLE_DEVICES="" python -m mlframe.training.composite.transforms._benchmarks.bench_grouped_fit_segment
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.training.composite.transforms.linear import _linear_residual_fit


def _old_pregather(y, base, inverse_idx, unique_groups, min_group_size):
    """OLD: O(K*n) per-group boolean mask + fancy-index gather, then lstsq."""
    out = {}
    for i, g in enumerate(unique_groups):
        g_mask = (inverse_idx == i)
        n_g = int(g_mask.sum())
        if n_g < min_group_size:
            continue
        y_g = y[g_mask]
        base_g = base[g_mask]
        p = _linear_residual_fit(y_g, base_g)
        out[int(g)] = (float(p["alpha"]), float(p["beta"]), n_g)
    return out


def _new_segment(y, base, inverse_idx, unique_groups, min_group_size):
    """NEW: one stable argsort + contiguous segment slices, then lstsq."""
    K = unique_groups.size
    order = np.argsort(inverse_idx, kind="stable")
    y_s = y[order]
    base_s = base[order]
    counts = np.bincount(inverse_idx, minlength=K)
    offsets = np.empty(K + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    out = {}
    for i in range(K):
        n_g = int(counts[i])
        if n_g < min_group_size:
            continue
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        y_g = y_s[lo:hi]
        base_g = base_s[lo:hi]
        p = _linear_residual_fit(y_g, base_g)
        out[int(unique_groups[i])] = (float(p["alpha"]), float(p["beta"]), n_g)
    return out


def _make(n, K, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, K, size=n)
    base = rng.standard_normal(n)
    y = 1.5 * base + 0.3 + 0.1 * rng.standard_normal(n)
    unique_groups, inverse_idx = np.unique(groups, return_inverse=True)
    return y, base, inverse_idx, unique_groups


def _best_of(fn, *args, reps=7):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, r


def main():
    min_group_size = 30
    print(f"{'n':>9} {'K':>6} {'old_ms':>9} {'new_ms':>9} {'speedup':>8} identical")
    for n, K in [(50_000, 500), (100_000, 1000), (200_000, 2000), (200_000, 5000)]:
        y, base, inv, uniq = _make(n, K)
        # warm
        _old_pregather(y, base, inv, uniq, min_group_size)
        _new_segment(y, base, inv, uniq, min_group_size)
        t_old, r_old = _best_of(_old_pregather, y, base, inv, uniq, min_group_size)
        t_new, r_new = _best_of(_new_segment, y, base, inv, uniq, min_group_size)
        ident = r_old == r_new  # exact float equality of (alpha, beta, n) per group
        print(f"{n:>9} {K:>6} {t_old*1e3:>9.2f} {t_new*1e3:>9.2f} "
              f"{t_old/t_new:>8.2f} {ident}")


if __name__ == "__main__":
    main()
