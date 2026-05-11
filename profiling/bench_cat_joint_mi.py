"""A/B bench: 3-pass vs single-pass joint-MI in the cat-interaction
permutation confirmation kernel.

Wave 10b 1M-row regression+MRMR profile (c0089) attributed 26.5 s
tottime to ``_count_nfailed_joint_indep_prange`` over 56 calls. The
pre-fix kernel called ``compute_mi_from_classes`` three times per
permutation iteration -- each call iterates ``zip(classes_x,
classes_y)`` over all N rows to build a joint-counts matrix. Single-
pass form fills three joint-counts in one iteration over N.

Usage:
    python -m mlframe.profiling.bench_cat_joint_mi
"""

from __future__ import annotations

import math
import statistics
import time

import numpy as np
import numba
from numba import njit, prange

from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes
from mlframe.feature_selection.filters.cat_interactions import (
    _count_nfailed_joint_indep_prange,
)


@njit(parallel=True, cache=False)
def _old_count_nfailed(
    classes_pair: np.ndarray,
    freqs_pair: np.ndarray,
    classes_x1: np.ndarray,
    freqs_x1: np.ndarray,
    classes_x2: np.ndarray,
    freqs_x2: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    ii_obs: float,
    n_perms: int,
    base_seed: int,
    dtype,
) -> int:
    """Pre-Wave-10 form: 3 separate compute_mi_from_classes per
    permutation. Inlined here so we can A/B-compare without monkey-
    patching production code."""
    n = len(classes_y)
    nfailed_total = 0
    for tid in prange(n_perms):
        cy_local = classes_y.copy()
        np.random.seed(base_seed + tid)
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            tmp = cy_local[i]
            cy_local[i] = cy_local[j]
            cy_local[j] = tmp
        i_pair = compute_mi_from_classes(
            classes_x=classes_pair, freqs_x=freqs_pair,
            classes_y=cy_local, freqs_y=freqs_y, dtype=dtype,
        )
        i_x1 = compute_mi_from_classes(
            classes_x=classes_x1, freqs_x=freqs_x1,
            classes_y=cy_local, freqs_y=freqs_y, dtype=dtype,
        )
        i_x2 = compute_mi_from_classes(
            classes_x=classes_x2, freqs_x=freqs_x2,
            classes_y=cy_local, freqs_y=freqs_y, dtype=dtype,
        )
        if (i_pair - i_x1 - i_x2) >= ii_obs:
            nfailed_total += 1
    return nfailed_total


def main() -> None:
    rng = np.random.default_rng(0)
    N = 1_000_000
    K_pair = 20
    K_x1 = 8
    K_x2 = 8
    K_y = 4

    classes_pair = rng.integers(0, K_pair, N).astype(np.int32)
    classes_x1 = rng.integers(0, K_x1, N).astype(np.int32)
    classes_x2 = rng.integers(0, K_x2, N).astype(np.int32)
    classes_y = rng.integers(0, K_y, N).astype(np.int32)

    freqs_pair = np.bincount(classes_pair, minlength=K_pair).astype(np.float64) / N
    freqs_x1 = np.bincount(classes_x1, minlength=K_x1).astype(np.float64) / N
    freqs_x2 = np.bincount(classes_x2, minlength=K_x2).astype(np.float64) / N
    freqs_y = np.bincount(classes_y, minlength=K_y).astype(np.float64) / N

    n_perms = 3
    base_seed = 7
    dtype = np.int32

    # Warm both kernels (numba JIT compile)
    _old_count_nfailed(
        classes_pair, freqs_pair, classes_x1, freqs_x1,
        classes_x2, freqs_x2, classes_y, freqs_y,
        0.0, n_perms, base_seed, dtype,
    )
    _count_nfailed_joint_indep_prange(
        classes_pair, freqs_pair, classes_x1, freqs_x1,
        classes_x2, freqs_x2, classes_y, freqs_y,
        0.0, n_perms, base_seed, dtype,
    )

    def bench(fn, label, n_repeat=5):
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        m = statistics.mean(times)
        s = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"  {label:<55} {m*1000:>8.1f} ms +/- {s*1000:>5.1f} ms")

    def _run_old():
        _old_count_nfailed(
            classes_pair, freqs_pair, classes_x1, freqs_x1,
            classes_x2, freqs_x2, classes_y, freqs_y,
            0.0, n_perms, base_seed, dtype,
        )

    def _run_new():
        _count_nfailed_joint_indep_prange(
            classes_pair, freqs_pair, classes_x1, freqs_x1,
            classes_x2, freqs_x2, classes_y, freqs_y,
            0.0, n_perms, base_seed, dtype,
        )

    print(f"# _count_nfailed_joint_indep_prange (N={N:_}, n_perms={n_perms})")
    bench(_run_old, "OLD: 3 passes (3x compute_mi_from_classes)")
    bench(_run_new, "NEW: single-pass joint-counts")


if __name__ == "__main__":
    main()
