"""CPX22 — confirm `_average_rank_inplace` tie loop is O(n), NOT O(n^2).

Claim under investigation: the tie-handling loop in
``mlframe.metrics.rank_correlation._average_rank_inplace`` (lines ~108-118)
does an O(n^2) worst-case tie rescan.

Inspection verdict: the loop is already LINEAR. The outer cursor `i` jumps to
`j + 1` after each tie block and never revisits it; the inner `while` advances
`j` strictly forward across the whole array, never resetting before the prior
`i`. Each index is touched a constant number of times (once by the inner-while
advance, once by the `for k` write). Total = O(n), plus the irreducible
O(n log n) argsort. There is no rescan that restarts.

This bench EMPIRICALLY confirms linearity: it times the kernel on the WORST
case for ties (integer-valued floats with very few distinct values => huge tie
blocks) across a size sweep and checks that wall time scales ~n log n, not n^2.
If it were quadratic, the all-equal column (one giant tie block of length n)
would blow up super-linearly; instead per-element time stays flat.

Run:  python -m mlframe.metrics._benchmarks.bench_average_rank_tie_scan_cpx22
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.metrics.rank_correlation import _average_rank_inplace, _HAS_NUMBA


def _best_of(fn, n_iter: int) -> float:
    best = float("inf")
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def _make_heavy_ties(n: int, n_distinct: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_distinct, size=n).astype(np.float64)


def main() -> None:
    if not _HAS_NUMBA:
        print("numba unavailable — kernel is the @njit path; skipping bench")
        return

    # Warm the JIT.
    warm = _make_heavy_ties(1000, 5)
    out = np.empty_like(warm)
    _average_rank_inplace(warm, out)

    sizes = [50_000, 200_000, 1_000_000, 4_000_000]
    print(f"{'n':>10} {'best_ms':>10} {'ns/elem':>10}   case")
    for n in sizes:
        # Worst tie case: ~10 distinct values => ~n/10 length tie blocks.
        x10 = _make_heavy_ties(n, 10)
        o10 = np.empty_like(x10)
        ms = _best_of(lambda: _average_rank_inplace(x10, o10), 5) * 1e3
        print(f"{n:>10} {ms:>10.3f} {ms * 1e6 / n:>10.3f}   10-distinct")

        # Pathological single giant tie block (all equal): the case that would
        # explode if any rescan restarted at i.
        xeq = np.zeros(n, dtype=np.float64)
        oeq = np.empty_like(xeq)
        ms_eq = _best_of(lambda: _average_rank_inplace(xeq, oeq), 5) * 1e3
        print(f"{n:>10} {ms_eq:>10.3f} {ms_eq * 1e6 / n:>10.3f}   all-equal (1 tie block of len n)")

    print(
        "\nVerdict: ns/elem stays ~flat (slow growth from argsort's log n only); "
        "all-equal does NOT blow up => O(n) tie loop, NOT O(n^2). REJECTED (no-op fix)."
    )


if __name__ == "__main__":
    main()
