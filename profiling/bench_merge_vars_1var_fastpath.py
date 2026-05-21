"""Bench 1-var fast-path for merge_vars in info_theory.py.

bench-attempt-rejected (2026-05-21, c0148 / iter136): the 1-var case
of merge_vars reduces to bincount + cast (final_classes[row] starts
at 0, current_nclasses=1, so newclass = sample_class). Replacing the
per-row loop with np.bincount + cast runs 15-20% SLOWER:

    n=  50000  K=  8: current=  0.086ms  fast=  0.107ms  (0.80x)
    n= 200000  K=  8: current=  0.383ms  fast=  0.451ms  (0.85x)
    n= 200000  K= 16: current=  0.378ms  fast=  0.448ms  (0.84x)
    n=1000000  K=  8: current=  4.165ms  fast=  4.894ms  (0.85x)
    n=1000000  K= 32: current=  4.539ms  fast=  4.625ms  (0.98x)

Cause: numba @njit already compiles the per-row loop into tight
machine code. The bincount path pays for (1) bincount returning int64
(extra cast back to dtype), (2) an extra .copy() to mirror the
legacy fresh-array contract on the no-zero-bin branch. The savings
of replacing the manual loop with bincount don't exceed those extra
costs.

Numerically bit-identical (eq=cls:True freq:True nc:True at every
size). The fastpath is correct, just slower.

c0148 attributed 2.72 s self-time across 358 calls of merge_vars
(~7.6 ms / call); the 1-var subset (~2/3 of calls) is ~1.8 s and
would have INCREASED by ~0.27 s with this rewrite.

Documented per ``feedback_document_failed_optimization_attempts`` so
the next agent doesn't re-try this same path.

Run: ``python profiling/bench_merge_vars_1var_fastpath.py``
"""

import time
import numpy as np
from numba import njit


@njit(cache=True)
def _merge_vars_current(factors_data, var_index, nbins_v, dtype):
    """Inlined 1-var path of merge_vars (mirrors the legacy inner loop)."""
    n = factors_data.shape[0]
    final_classes = np.zeros(n, dtype=dtype)
    freqs = np.zeros(nbins_v, dtype=dtype)
    values = factors_data[:, var_index].astype(dtype)
    for sample_row in range(n):
        sample_class = values[sample_row]
        # current_nclasses=1, final_classes[row]=0 -> newclass = sample_class
        newclass = sample_class
        freqs[newclass] += 1
        final_classes[sample_row] = newclass
    # Zero-bin prune
    nzeros = 0
    lookup_table = np.empty(nbins_v, dtype=dtype)
    for oldclass in range(nbins_v):
        if freqs[oldclass] == 0:
            nzeros += 1
        lookup_table[oldclass] = oldclass - nzeros
    if nzeros:
        for sample_row in range(n):
            final_classes[sample_row] = lookup_table[final_classes[sample_row]]
        freqs = freqs[freqs > 0]
    current_nclasses = nbins_v - nzeros
    return final_classes, freqs / n, current_nclasses


@njit(cache=True)
def _merge_vars_1var_fast(factors_data, var_index, nbins_v, dtype):
    """1-var fast-path: replace the per-row loop with bincount + cast."""
    n = factors_data.shape[0]
    # Cast column to dtype (single pass; numba inlines)
    values = factors_data[:, var_index].astype(dtype)
    # Bincount-equivalent: numba supports np.bincount in nopython mode.
    freqs = np.bincount(values, minlength=nbins_v)
    # The current path stores freqs in `dtype`. np.bincount returns int64;
    # cast back for compatibility.
    freqs_dt = freqs.astype(dtype)
    # Zero-bin prune (same logic as the legacy path)
    nzeros = 0
    lookup_table = np.empty(nbins_v, dtype=dtype)
    for oldclass in range(nbins_v):
        if freqs_dt[oldclass] == 0:
            nzeros += 1
        lookup_table[oldclass] = oldclass - nzeros
    if nzeros:
        # Lookup-remap final_classes
        final_classes = np.empty(n, dtype=dtype)
        for sample_row in range(n):
            final_classes[sample_row] = lookup_table[values[sample_row]]
        freqs_dt = freqs_dt[freqs_dt > 0]
    else:
        # No prune needed; the dtype-cast `values` IS the final class array.
        # But the legacy path returns a fresh array; mimic via copy.
        final_classes = values.copy()
    current_nclasses = nbins_v - nzeros
    return final_classes, freqs_dt / n, current_nclasses


def bench(label, fn, args, n_iter=200):
    fn(*args); fn(*args)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(*args)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e3, label


if __name__ == "__main__":
    for n, K in [(50_000, 8), (200_000, 8), (200_000, 16), (1_000_000, 8), (1_000_000, 32)]:
        rng = np.random.default_rng(0)
        factors_data = rng.integers(0, K, size=(n, 4), dtype=np.int32)
        args_current = (factors_data, 1, K, np.int32)
        args_fast = (factors_data, 1, K, np.int32)
        t_cur, _ = bench("current", _merge_vars_current, args_current)
        t_fast, _ = bench("fast",    _merge_vars_1var_fast, args_fast)
        # Verify equivalence
        a, b, c = _merge_vars_current(*args_current)
        d, e, f = _merge_vars_1var_fast(*args_fast)
        eq_classes = np.array_equal(a, d)
        eq_freqs = np.allclose(b, e)
        eq_nclasses = (c == f)
        speedup = t_cur / t_fast
        print(f"n={n:>7}  K={K:>3}: current={t_cur:7.3f}ms  fast={t_fast:7.3f}ms  ({speedup:.2f}x)  eq=cls:{eq_classes} freq:{eq_freqs} nc:{eq_nclasses}")
