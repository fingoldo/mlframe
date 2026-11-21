"""Bench CPX39: MyDecorrelator.fit O(p^2) double-loop vs vectorized np.triu(k=1).

Run: CUDA_VISIBLE_DEVICES="" python src/mlframe/estimators/_benchmarks/bench_cpx39_decorrelator.py

The NEW side is the production class ``MyDecorrelator(threshold).fit(X)``. The OLD side is a replica of the
pre-vectorisation double loop, kept here as the reference the identity gate compares against. It is not loaded from git
history: this bench does not read an old revision of custom.py, so its OLD timing describes the replica, and a change to
the production class shows up in the NEW column and in the identity check (audit 06-F1, 06-F2, 06-F3).
Warm + best-of-N; identity gate on ``correlated_features_`` across several seeds.
"""

import numpy as np, pandas as pd

from mlframe._bench_timing_shared import best_of_seconds
from mlframe.estimators.custom import MyDecorrelator


def _make_data(n=5000, p=200, n_corr_pairs=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # Inject correlated columns: make col j a near-copy of an earlier col i.
    for _k in range(n_corr_pairs):
        i = rng.integers(0, p // 2)
        j = rng.integers(p // 2, p)
        X[:, j] = X[:, i] + 0.05 * rng.standard_normal(n)
    return pd.DataFrame(X)


def _old_fit(X, threshold):
    """Replica of the pre-vectorisation loop: a column is flagged when an EARLIER column correlates above threshold. Returns integer positions."""
    corr_matrix = pd.DataFrame(X).corr()
    flagged = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                flagged.add(i)
    return flagged


def _new_fit(X, threshold):
    """The production class's fit: its ``correlated_features_`` (integer positions)."""
    return set(MyDecorrelator(threshold).fit(X).correlated_features_)


_best_of = best_of_seconds


def main():
    old_fit = _old_fit
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
        print(f"{f'(5000,200) s{seed}':>12} {old_ms:>10.2f} {new_ms:>10.2f} {old_ms/new_ms:>7.2f}x {ident!s:>9}")
    print(f"\nALL identical across seeds: {all_ident}")


if __name__ == "__main__":
    main()
