"""Bench: njit single-pass scatter for ``_build_factorize_lookup`` vs the
current numpy form (two ``.astype(int64)`` column copies + a length-n
``vals_a + vals_b * nbins_a`` temporary + fancy-index scatter).

The current numpy population in ``_cat_kway_materialize._build_factorize_lookup``
allocates THREE length-n int64 temporaries per call (vals_a, vals_b,
pre_prune_codes) before the fancy-index scatter into ``lookup``. This runs once
per selected pair (``CatFEConfig.top_k_pairs`` default 32) per fit. An njit
kernel reads the two columns directly and scatters the post-prune class into
``lookup`` in a single pass with NO temporaries -- bit-identical by construction
(same ``code = a + b * nbins_a`` integer arithmetic, last-write-wins on dup
codes exactly as numpy fancy-index assignment, which assigns in row order).

Run:
    CUDA_VISIBLE_DEVICES="" python bench_factorize_lookup_njit_scatter.py
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit


# ---- OLD: numpy population (verbatim from _build_factorize_lookup, pre-fix) ----
def _populate_numpy(factors_data, idx_a, idx_b, nbins_a, nbins_b, classes_pair_post):
    expected_size = int(nbins_a) * int(nbins_b)
    lookup = np.full(expected_size, -1, dtype=np.int64)
    vals_a = factors_data[:, idx_a].astype(np.int64)
    vals_b = factors_data[:, idx_b].astype(np.int64)
    pre_prune_codes = vals_a + vals_b * int(nbins_a)
    lookup[pre_prune_codes] = classes_pair_post.astype(np.int64)
    return lookup


# ---- NEW: njit single-pass scatter, no length-n temporaries ----
@njit(cache=True)
def _populate_njit(factors_data, idx_a, idx_b, nbins_a, nbins_b, classes_pair_post):
    expected_size = nbins_a * nbins_b
    lookup = np.full(expected_size, -1, dtype=np.int64)
    n = factors_data.shape[0]
    for r in range(n):
        code = factors_data[r, idx_a] + factors_data[r, idx_b] * nbins_a
        lookup[code] = classes_pair_post[r]
    return lookup


def _make(n, nbins_a, nbins_b, seed):
    rng = np.random.default_rng(seed)
    fd = np.empty((n, 3), dtype=np.int32)
    fd[:, 0] = rng.integers(0, nbins_a, n)
    fd[:, 1] = rng.integers(0, nbins_b, n)
    fd[:, 2] = rng.integers(0, 5, n)
    # post-prune classes: dense renumber of the seen (a,b) pairs in row order
    codes = fd[:, 0].astype(np.int64) + fd[:, 1].astype(np.int64) * nbins_a
    classes = np.empty(n, dtype=np.int32)
    seen = {}
    nxt = 0
    for r in range(n):
        c = int(codes[r])
        if c not in seen:
            seen[c] = nxt
            nxt += 1
        classes[r] = seen[c]
    return fd, classes


def best_of(fn, n_iter=30):
    best = 1e18
    for _ in range(n_iter):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


if __name__ == "__main__":
    print(f"{'n':>9} {'nb_a':>5} {'nb_b':>5} {'old_us':>9} {'new_us':>9} {'speedup':>8} identical")
    top_k = 32
    for n in (10_000, 50_000, 100_000, 200_000):
        for nba, nbb in ((10, 10), (20, 16)):
            fd, cls = _make(n, nba, nbb, seed=n + nba)
            old = _populate_numpy(fd, 0, 1, nba, nbb, cls)
            new = _populate_njit(fd, 0, 1, nba, nbb, cls)
            ident = np.array_equal(old, new)
            # warm njit already done above
            # simulate top_k_pairs calls per fit
            t_old = best_of(lambda: [_populate_numpy(fd, 0, 1, nba, nbb, cls) for _ in range(top_k)], 15)
            t_new = best_of(lambda: [_populate_njit(fd, 0, 1, nba, nbb, cls) for _ in range(top_k)], 15)
            print(f"{n:>9} {nba:>5} {nbb:>5} {t_old*1e6:>9.1f} {t_new*1e6:>9.1f} " f"{t_old/t_new:>7.2f}x {ident}")
