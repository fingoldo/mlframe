"""Bench the cross-stage engineered-column dedup scan's correlation kernel.

The scan compares each newly-kept column against every column kept before it, so the cost is O(K^2) comparisons over the ~200 engineered
columns a wide fit can produce. Run from the repository root::

    python -m mlframe.feature_selection._benchmarks.bench_eng_dedup_batch_corr

Simulates the whole scan rather than a single call, because the saving is per comparison and only the scan shows the O(K^2) shape.

bench-attempt-rejected (2026-07-13): a naive per-candidate ``np.vstack`` of the currently-kept rank vectors measured 0.88x at 200 columns /
100k rows, a net loss, because it re-copies O(K) rows of length n on every candidate. The append-only buffer plus active mask replaced it.
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_batch_corr import (
    one_vs_many_abs_corr_masked,
    row_mean_and_centred_ss,
)

_K_GRID = (20, 50, 100, 200)
_N_GRID = (10_000, 100_000)


def _scan_without_cached_moments(rows: np.ndarray) -> float:
    """The whole dedup scan with each row's mean and centred sum-of-squares recomputed per comparison."""
    total = 0.0
    k = rows.shape[0]
    for j in range(1, k):
        active = np.ones(j, dtype=np.bool_)
        total += float(one_vs_many_abs_corr_masked(rows[j], rows[:j], active).sum())
    return total


def _scan_with_cached_moments(rows: np.ndarray) -> float:
    """The same scan with each row's moments computed once, when the row is appended."""
    total = 0.0
    k = rows.shape[0]
    means = np.empty(k, dtype=np.float64)
    sss = np.empty(k, dtype=np.float64)
    means[0], sss[0] = row_mean_and_centred_ss(rows[0])
    for j in range(1, k):
        means[j], sss[j] = row_mean_and_centred_ss(rows[j])
        active = np.ones(j, dtype=np.bool_)
        total += float(one_vs_many_abs_corr_masked(rows[j], rows[:j], active, means[:j], sss[:j]).sum())
    return total


def _best(fn, rows: np.ndarray, repeat: int) -> float:
    """Best-of-``repeat`` wall seconds, warmed once so no run pays compilation."""
    fn(rows)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(rows)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    """Run the sweep and print the per-scan speedup with the agreement of the two forms."""
    print(f"{'n':>8} {'K':>5} {'pairs':>7} {'recompute s':>12} {'cached s':>10} {'speedup':>8} {'rel diff':>10}")
    for n in _N_GRID:
        for k in _K_GRID:
            rng = np.random.default_rng(0)
            rows = rng.normal(size=(k, n))
            repeat = 3 if n <= 10_000 else 2
            t_recompute = _best(_scan_without_cached_moments, rows, repeat)
            t_cached = _best(_scan_with_cached_moments, rows, repeat)
            a, b = _scan_without_cached_moments(rows), _scan_with_cached_moments(rows)
            rel = abs(a - b) / max(abs(a), 1e-12)
            print(f"{n:>8} {k:>5} {k * (k - 1) // 2:>7} {t_recompute:>12.4f} {t_cached:>10.4f} {t_recompute / t_cached:>7.2f}x {rel:>10.2e}")


if __name__ == "__main__":
    main()
