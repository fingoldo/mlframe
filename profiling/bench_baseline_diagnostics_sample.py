"""Bench baseline_diagnostics._sample at the production call shape.

c0052 iter168 profile attributed 322ms / 1 call to _sample at n=100k.
Default config.sample_n=50_000. Hot path:
    idx = rng.choice(n, size=sample_n, replace=False)
    idx.sort()
    X.iloc[idx].reset_index(drop=True)
    y[idx]

The .choice(replace=False) generates a full O(n) permutation internally
(numpy uses Fisher-Yates). For sample_n << n / 2, Floyd's algorithm
generates sample_n uniques in O(sample_n) without touching the full
n-space. But sample_n=50k of n=100k is HALF -- not sample_n << n. So
Floyd's may not win.

Run: ``python profiling/bench_baseline_diagnostics_sample.py``
"""
import time
import numpy as np
import pandas as pd


def sample_choice(X, y, sample_n, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, size=sample_n, replace=False)
    idx.sort()
    return X.iloc[idx].reset_index(drop=True), y[idx], sample_n


def sample_permutation(X, y, sample_n, seed=0):
    """Use np.random.permutation + slice -- same complexity but skips
    the size-check + replace=False guard."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)[:sample_n]
    idx.sort()
    return X.iloc[idx].reset_index(drop=True), y[idx], sample_n


def sample_floyd(X, y, sample_n, seed=0):
    """Floyd's algorithm: build the sample set incrementally."""
    rng = np.random.default_rng(seed)
    n = len(X)
    selected = set()
    # Standard Floyd: for j in [n-sample_n, n-1]: pick uniform [0, j], add j or random pick.
    for j in range(n - sample_n, n):
        t = int(rng.integers(0, j + 1))
        if t in selected:
            selected.add(j)
        else:
            selected.add(t)
    idx = np.array(sorted(selected), dtype=np.int64)
    return X.iloc[idx].reset_index(drop=True), y[idx], sample_n


def bench(label, fn, X, y, sample_n, n_iter=20):
    fn(X, y, sample_n); fn(X, y, sample_n)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(X, y, sample_n)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e3, label


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n, K in [(100_000, 10), (100_000, 30), (500_000, 10)]:
        X = pd.DataFrame(rng.standard_normal((n, K)))
        y = rng.standard_normal(n)
        for sample_n in (50_000,):
            if sample_n >= n:
                continue
            t_c, _ = bench("choice", sample_choice, X, y, sample_n)
            t_p, _ = bench("perm", sample_permutation, X, y, sample_n)
            # Skip Floyd at large sample_n (Python set ops too slow)
            print(f"n={n:>7} K={K:>2} sample_n={sample_n}: "
                  f"choice={t_c:6.1f}ms perm={t_p:6.1f}ms  ({t_c/t_p:.2f}x)")
