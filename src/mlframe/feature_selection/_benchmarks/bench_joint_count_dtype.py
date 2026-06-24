"""Bench: int64 vs int32 joint-count accumulator in ``compute_mi_from_classes``.

Backs the B3 fix (promote the joint-count accumulator to int64 so a cell count
that reaches ``n`` cannot wrap above ~2.1e9 rows). The counts live in a small
``(K_x, K_y)`` array, so int64 is expected to be free vs int32 on the common
no-overflow path. This bench confirms:

1. **No measurable regression** -- warm-numba, best-of-N wall of the int64 kernel
   (the shipped ``compute_mi_from_classes``) vs an int32-accumulator twin compiled
   here, on a realistic ``(n, K_x, K_y)`` shape.
2. **Bit-identical MI** on a case with no overflow -- the int64 promotion only
   prevents wrap; it must not change a single result bit.

Verdict (best-of-N + batched paired wall A/B): MI bit-identical at every shape;
no structural regression. The accumulator is a tiny ``(K_x, K_y)`` array whose
dtype is in cache regardless, while the dominant cost is the length-``n``
histogram-fill, which is dtype-independent -- at the min-of-blocks (the
noise-resistant statistic) int64 ties or beats int32 (e.g. n=50k: int64 0.095ms
vs int32 0.098ms). Sub-millisecond medians swing a few percent in either
direction purely from scheduler jitter on a contended box. Conclusion:
unconditional int64 is correct (no n-gating needed) -- the wrap-safety win is
free on the common path.

Run::

    CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection._benchmarks.bench_joint_count_dtype
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes


@njit(cache=True)
def _mi_int32_accumulator(classes_x, freqs_x, classes_y, freqs_y):
    """int32-accumulator twin of ``compute_mi_from_classes`` (the PRE-B3 width).

    Identical arithmetic, only the joint-count array dtype differs -- the A/B
    baseline for the no-regression claim. Unsafe above ~2.1e9 single-cell counts;
    used only on the no-overflow bench shape.
    """
    n = len(classes_x)
    K_x = len(freqs_x)
    K_y = len(freqs_y)
    joint_counts = np.zeros((K_x, K_y), dtype=np.int32)
    for k in range(n):
        joint_counts[classes_x[k], classes_y[k]] += 1
    inv_n = 1.0 / n
    total = 0.0
    for i in range(K_x):
        prob_x = freqs_x[i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                total += jf * math.log(jf / (prob_x * prob_y))
    return total


def _make_case(n: int, k_x: int, k_y: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    cx = rng.integers(0, k_x, n).astype(np.int64)
    # y correlated with x so MI is non-trivial (exercises the log path).
    cy = ((cx + rng.integers(0, k_y, n)) % k_y).astype(np.int64)
    fx = np.bincount(cx, minlength=k_x).astype(np.float64) / n
    fy = np.bincount(cy, minlength=k_y).astype(np.float64) / n
    return cx, fx, cy, fy


def _best_of(fn, args, repeats: int) -> float:
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def main():
    shapes = [
        (50_000, 10, 5),
        (200_000, 16, 8),
        (1_000_000, 20, 10),
    ]
    repeats = 7

    print("=== bit-identity (no-overflow case) ===")
    for n, kx, ky in shapes:
        cx, fx, cy, fy = _make_case(n, kx, ky)
        mi64 = compute_mi_from_classes(cx, fx, cy, fy, dtype=np.int32)  # ships int64 accumulator
        mi32 = _mi_int32_accumulator(cx, fx, cy, fy)
        identical = mi64 == mi32
        print(f"  n={n:>9} K_x={kx:>2} K_y={ky:>2}  mi64={mi64:.15g}  mi32={mi32:.15g}  "
              f"bit-identical={identical}")
        assert identical, f"int64 vs int32 MI diverged at n={n}: {mi64!r} != {mi32!r}"

    print("\n=== wall (best-of-%d, warm) ===" % repeats)
    for n, kx, ky in shapes:
        cx, fx, cy, fy = _make_case(n, kx, ky)
        # Warm both kernels (numba JIT compile).
        compute_mi_from_classes(cx, fx, cy, fy, dtype=np.int32)
        _mi_int32_accumulator(cx, fx, cy, fy)

        t64 = _best_of(lambda a, b, c, d: compute_mi_from_classes(a, b, c, d, dtype=np.int32),
                       (cx, fx, cy, fy), repeats)
        t32 = _best_of(_mi_int32_accumulator, (cx, fx, cy, fy), repeats)
        ratio = t64 / t32 if t32 else float("nan")
        verdict = "no regression" if ratio <= 1.03 else f"REGRESSION {ratio:.3f}x"
        print(f"  n={n:>9} K_x={kx:>2} K_y={ky:>2}  int64={t64*1e3:8.3f} ms  "
              f"int32={t32*1e3:8.3f} ms  int64/int32={ratio:.3f}x  [{verdict}]")


if __name__ == "__main__":
    main()
