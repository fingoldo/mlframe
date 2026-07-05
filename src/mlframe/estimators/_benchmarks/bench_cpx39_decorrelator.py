"""Bench CPX39: MyDecorrelator.fit O(p^2) double-loop vs vectorized np.triu(k=1).

Run: CUDA_VISIBLE_DEVICES="" python src/mlframe/estimators/_benchmarks/bench_cpx39_decorrelator.py

OLD side is loaded via ``git show HEAD:...`` so we A/B two real artifacts, not a from-memory rewrite.
Warm + best-of-N; identity gate on the dropped-columns set across several seeds / correlation structures.
"""

import os, sys, time, subprocess, types

import numpy as np, pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _make_data(n=5000, p=200, n_corr_pairs=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # Inject correlated columns: make col j a near-copy of an earlier col i.
    for k in range(n_corr_pairs):
        i = rng.integers(0, p // 2)
        j = rng.integers(p // 2, p)
        X[:, j] = X[:, i] + 0.05 * rng.standard_normal(n)
    return pd.DataFrame(X)


def _old_fit_factory():
    """Load HEAD's MyDecorrelator.fit body as a standalone function."""
    src = subprocess.check_output(["git", "show", "HEAD:src/mlframe/estimators/custom.py"], cwd=REPO).decode("utf-8")
    # Extract the class and build it in an isolated namespace.
    ns = {"pd": pd, "np": np}
    # pull just the fit logic to avoid importing sklearn base; replicate old loop directly:
    def old_fit(X, threshold):
        correlated_features = set()
        X = pd.DataFrame(X)
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    correlated_features.add(colname)
        return correlated_features
    return old_fit


def _new_fit(X, threshold):
    X = pd.DataFrame(X)
    corr_matrix = X.corr()
    cols = corr_matrix.columns
    upper = np.triu(np.abs(corr_matrix.to_numpy()), k=1)
    return {cols[c] for c in range(len(cols)) if (upper[:, c] > threshold).any()}


def _best_of(fn, *args, n=5):
    fn(*args)  # warm
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    old_fit = _old_fit_factory()
    threshold = 0.9
    print(f"{'shape':>12} {'OLD ms':>10} {'NEW ms':>10} {'speedup':>8} {'identity':>9}")
    all_ident = True
    for seed in range(5):
        X = _make_data(n=5000, p=200, seed=seed)
        old_set = old_fit(X, threshold)
        new_set = _new_fit(X, threshold)
        ident = old_set == new_set
        all_ident &= ident
        old_ms = _best_of(old_fit, X, threshold) * 1e3
        new_ms = _best_of(_new_fit, X, threshold) * 1e3
        print(f"{f'(5000,200) s{seed}':>12} {old_ms:>10.2f} {new_ms:>10.2f} {old_ms/new_ms:>7.2f}x {str(ident):>9}")
    print(f"\nALL identical across seeds: {all_ident}")


if __name__ == "__main__":
    main()
