"""Permutation null for the offset grid (CPU njit, deterministic across thread counts): bin all K*G candidates ONCE (uint8 codes), then B shuffled-y histogram passes.
Null statistic per permutation = max over all K*G candidates (covers the selection over k and g)."""

import time

import numba
import numpy as np
from numba import njit, prange

from .offset_kernels import DEFAULT_FINE_BINS, _mi_from_hist, candidate_codes_b, score_offset_grid_b

NULL_SEED_MIX = 0x9E3779B97F4A7C15


@njit(parallel=True, nogil=True, cache=True)
def build_codes(U, V, T, nbins, NB):
    """Bin every K*G candidate once (uint8 codes) with the exact variant-b edges."""
    K, n = U.shape
    G = T.shape[1]
    codes = np.empty((K * G, n), dtype=np.uint8)
    for c in prange(K * G):
        buf = np.empty(n, dtype=np.int8)
        candidate_codes_b(U[c // G], V[c // G], T[c // G, c % G], nbins, NB, buf)
        for i in range(n):
            codes[c, i] = buf[i]
    return codes


@njit(cache=True, nogil=True)
def _xorshift(s):
    """One xorshift64 step (thread-count independent shuffle stream)."""
    s ^= (s << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    s ^= s >> np.uint64(7)
    s ^= (s << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return s


@njit(parallel=True, nogil=True, cache=True)
def null_max_over_grid(codes, y, ncls, nbins, B, seed):
    """Returns (B, C) permuted MI.  Permutation b is a Fisher-Yates of y driven by a xorshift stream seeded from (seed, b),
    so the result does not depend on the thread count."""
    C, n = codes.shape
    out = np.empty((B, C))
    for b in prange(B):
        yp = y.copy()
        s = np.uint64(seed) ^ (np.uint64(b + 1) * np.uint64(NULL_SEED_MIX))
        for _ in range(4):
            s = _xorshift(s)
        for i in range(n - 1, 0, -1):
            s = _xorshift(s)
            j = int(s % np.uint64(i + 1))
            tmp = yp[i]
            yp[i] = yp[j]
            yp[j] = tmp
        hxy = np.empty(nbins * ncls, dtype=np.int64)
        hx = np.empty(nbins, dtype=np.int64)
        hy = np.zeros(ncls, dtype=np.int64)
        for i in range(n):
            hy[yp[i]] += 1
        for c in range(C):
            hxy[:] = 0
            hx[:] = 0
            row = codes[c]
            for i in range(n):
                bb = row[i]
                hxy[bb * ncls + yp[i]] += 1
                hx[bb] += 1
            out[b, c] = _mi_from_hist(hxy, hx, hy, ncls, nbins, n)
    return out


if __name__ == "__main__":
    from ._synthetic import NB, NCLS, make

    for n in (20_000, 100_000):
        U, V, T, yc = make(n)
        build_codes(U[:, :3000].copy(), V[:, :3000].copy(), T, NB, DEFAULT_FINE_BINS)
        null_max_over_grid(np.zeros((2, 3000), np.uint8), yc[:3000], NCLS, NB, 2, 1)
        t0 = time.perf_counter()
        codes = build_codes(U, V, T, NB, DEFAULT_FINE_BINS)
        tb = time.perf_counter() - t0
        res = {}
        for nt in (1, 8):
            numba.set_num_threads(nt)
            ts = []
            for _ in range(2):
                t0 = time.perf_counter()
                r = null_max_over_grid(codes, yc, NCLS, NB, 20, 12345)
                ts.append(time.perf_counter() - t0)
            res[nt] = r
            print(f"n={n} K*G=80 B=20 threads={nt}: bin-once {tb:.3f}s (8 thr) + shuffled-hist passes {min(ts):.3f}s")
        numba.set_num_threads(8)
        print(
            "   thread-count invariant:",
            np.array_equal(res[1], res[8]),
            " null max-over-grid mean/q95:",
            res[8].max(1).mean(),
            np.quantile(res[8].max(1), 0.95),
        )
        t0 = time.perf_counter()
        score_offset_grid_b(U, V, T, yc, NCLS, NB, DEFAULT_FINE_BINS)
        print(f"   (observed-grid scoring for scale: {time.perf_counter() - t0:.3f}s)")
