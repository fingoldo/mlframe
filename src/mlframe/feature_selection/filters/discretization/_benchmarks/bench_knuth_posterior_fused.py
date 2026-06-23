"""A/B bench + identity gate for the Knuth optimal-bin-count posterior search.

OLD path (``_knuth_bin_edges``): the M-loop calls ``np.histogram(a, bins=linspace(a_min, a_max, M+1))``
plus ``counts.astype(np.int64)`` for every M in [2, M_max], then an njit ``_knuth_log_posterior``.
Each iteration re-scans the whole array (a fresh ``np.histogram`` dispatch + an int64 copy alloc),
so the search is O(n * M_max) of object-mode numpy dispatch.

NEW path (``_knuth_best_M_fused``): sort ``a`` ONCE, then a single njit kernel walks M in [2, M_max]
computing the per-bin counts of UNIFORM-width bins by integer differencing of cumulative positions
(``np.searchsorted`` of the M+1 linspace edges into the sorted array, ``side='right'`` -- the exact
semantics ``np.histogram`` uses for its uniform fast path: half-open ``[e_i, e_{i+1})`` bins with the
final bin closed) and accumulates the lgamma log-posterior in compiled code -- no per-M numpy dispatch,
no int64 copy, no Python-level histogram object churn.

IDENTITY: best_M (hence the returned edges) is asserted bit-identical to the OLD path across
uniform / normal / heavy-tail / tie-heavy / skewed columns at n in {500, 2000, 10000, 50000}.

Run: CUDA_VISIBLE_DEVICES="" python bench_knuth_posterior_fused.py
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit


# ----------------------------------------------------------------------------- OLD (real prior code, copied verbatim)
@njit(nogil=True, cache=True)
def _knuth_log_posterior(M, n, counts):
    if M < 1 or n < 1:
        return -1e300
    log_M = math.log(M)
    log_gamma_half = math.lgamma(0.5)
    s = n * log_M + math.lgamma(M / 2.0) - M * log_gamma_half - math.lgamma(n + M / 2.0)
    for k in range(M):
        s += math.lgamma(counts[k] + 0.5)
    return s


def _knuth_best_M_old(a, m_max_cap=64):
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    a_min, a_max = float(a.min()), float(a.max())
    M_max = int(min(max(4, int(np.sqrt(n) * 4)), int(m_max_cap)))
    best_M, best_logp = 2, -1e300
    for M in range(2, M_max + 1):
        edges = np.linspace(a_min, a_max, M + 1)
        counts, _ = np.histogram(a, bins=edges)
        logp = _knuth_log_posterior(M, n, counts.astype(np.int64))
        if logp > best_logp:
            best_logp = logp
            best_M = M
    return best_M


# ----------------------------------------------------------------------------- NEW (fused njit kernel)
@njit(nogil=True, cache=True)
def _knuth_best_M_fused(a_sorted, a_min, a_max, M_max):
    n = a_sorted.shape[0]
    log_gamma_half = math.lgamma(0.5)
    best_M = 2
    best_logp = -1e300
    counts = np.empty(M_max, dtype=np.int64)
    for M in range(2, M_max + 1):
        # Uniform edges e_j = a_min + j*(a_max-a_min)/M; count via searchsorted on sorted data.
        # side='right' on the interior edges reproduces np.histogram's half-open [e_j, e_{j+1})
        # bins with the final bin closed: prev = #{x <= e_j}, count_j = #{x <= e_{j+1}} - prev,
        # and the final bin picks up x == a_max (== e_M) exactly as np.histogram does.
        width = (a_max - a_min) / M
        prev = 0
        # accumulate posterior inline
        s = n * math.log(M) + math.lgamma(M / 2.0) - M * log_gamma_half - math.lgamma(n + M / 2.0)
        for j in range(M):
            if j == M - 1:
                hi = n  # last bin closed at a_max: all remaining points
            else:
                edge = a_min + (j + 1) * width
                # number of points <= edge (side='right')
                hi = np.searchsorted(a_sorted, edge, side="right")
            c = hi - prev
            prev = hi
            counts[j] = c
            s += math.lgamma(c + 0.5)
        if s > best_logp:
            best_logp = s
            best_M = M
    return best_M


def _knuth_best_M_new(a, m_max_cap=64):
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    a_min, a_max = float(a.min()), float(a.max())
    M_max = int(min(max(4, int(np.sqrt(n) * 4)), int(m_max_cap)))
    a_sorted = np.sort(a)
    return _knuth_best_M_fused(a_sorted, a_min, a_max, M_max)


# ----------------------------------------------------------------------------- data
def make_cols(n, rng):
    return {
        "uniform": rng.uniform(0, 1, n),
        "normal": rng.normal(0, 1, n),
        "heavy_tail": rng.standard_t(2.0, n),
        "skewed": rng.exponential(1.0, n),
        "tie_heavy": rng.integers(0, 7, n).astype(np.float64),
        "lognormal": rng.lognormal(0, 1.0, n),
    }


def identity_check():
    rng = np.random.default_rng(12345)
    bad = 0
    total = 0
    for n in (500, 2000, 10000, 50000):
        for name, col in make_cols(n, rng).items():
            for cap in (64, 500):
                old = _knuth_best_M_old(col, cap)
                new = _knuth_best_M_new(col, cap)
                total += 1
                if old != new:
                    bad += 1
                    print(f"  MISMATCH n={n} col={name} cap={cap}: old={old} new={new}")
    print(f"identity: {total - bad}/{total} best_M bit-identical")
    return bad == 0


def bench():
    rng = np.random.default_rng(7)
    # warm njit
    _knuth_best_M_new(rng.normal(0, 1, 1000), 64)
    for n in (2000, 10000, 50000):
        cols = make_cols(n, rng)
        for name, col in cols.items():
            def t(fn, reps):
                best = 1e30
                for _ in range(reps):
                    t0 = time.perf_counter()
                    fn(col, 64)
                    best = min(best, time.perf_counter() - t0)
                return best
            reps = 20 if n <= 10000 else 8
            old = t(_knuth_best_M_old, reps)
            new = t(_knuth_best_M_new, reps)
            print(f"n={n:6d} {name:11s} OLD={old*1e3:8.3f}ms NEW={new*1e3:8.3f}ms  speedup={old/new:.2f}x")


if __name__ == "__main__":
    ok = identity_check()
    print("=" * 60)
    bench()
    print("=" * 60)
    print("IDENTITY OK" if ok else "IDENTITY FAILED")
