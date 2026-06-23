"""CPX12b bench: per-fold per-column train-edge binning vs batched-across-columns.

The OOF MI scorer in ``_orthogonal_three_gate_mi_fe.py`` bins every (fold x column)
pair with a scalar ``_bin_with_train_edges`` call: one ``np.quantile`` + two
``np.searchsorted`` per column per fold. Sibling scorers batch the quantile across
columns. The catch (documented FUTURE note): the OOF estimate must fit bin edges on
TRAIN rows only (leakage-free K-fold), so the existing full-frame batched kernels
are NOT a safe drop-in -- they quantile the WHOLE column (test rows leak into edges).

This bench measures the leakage-PRESERVING batched alternative
(``_bin_with_train_edges_batched``): for each fold, ``np.quantile(train_arr, qs,
axis=0)`` once for ALL columns, then per-column ``searchsorted`` applied to the test
rows -- identical edges to the scalar path, just vectorised across columns.

Run:
    python -m mlframe.feature_selection.filters._benchmarks.bench_oof_three_gate_train_edge_binning
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters._orthogonal_three_gate_mi_fe import (
    _bin_with_train_edges,
    _bin_with_train_edges_batched,
)


def _scalar_path(arr, fold_test_idx, nbins):
    """Reproduce the OLD per-(fold x column) scalar loop (test bins only)."""
    n, p = arr.shape
    out = []
    for test_idx in fold_test_idx:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        test_bins = np.empty((test_idx.size, p), dtype=np.int64)
        for j in range(p):
            _, tb = _bin_with_train_edges(
                arr[train_idx, j], arr[test_idx, j], nbins=nbins,
            )
            test_bins[:, j] = tb
        out.append(test_bins)
    return out


def _batched_path(arr, fold_test_idx, nbins):
    """The NEW batched-across-columns path (test bins only)."""
    n, p = arr.shape
    out = []
    for test_idx in fold_test_idx:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        _, test_bins = _bin_with_train_edges_batched(
            arr[train_idx, :], arr[test_idx, :], nbins=nbins,
        )
        out.append(test_bins)
    return out


def _make_folds(n, n_folds, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [perm[i::n_folds] for i in range(n_folds)]


def bench(n, p, n_folds, nbins=10, repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, p))
    # Inject a few low-cardinality / tied columns (fragile binning edges).
    arr[:, 0] = rng.integers(0, 3, size=n).astype(np.float64)
    if p > 5:
        arr[:, 5] = (arr[:, 5] > 0).astype(np.float64)
    folds = _make_folds(n, n_folds, seed)

    # Identity check first.
    old = _scalar_path(arr, folds, nbins)
    new = _batched_path(arr, folds, nbins)
    identical = all(np.array_equal(a, b) for a, b in zip(old, new))

    def _time(fn):
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(arr, folds, nbins)
            best = min(best, time.perf_counter() - t0)
        return best

    t_old = _time(_scalar_path)
    t_new = _time(_batched_path)
    print(
        f"n={n:>7} p={p:>4} folds={n_folds}  "
        f"OLD={t_old * 1e3:8.2f}ms  NEW={t_new * 1e3:8.2f}ms  "
        f"speedup={t_old / t_new:5.2f}x  identical={identical}"
    )
    return t_old, t_new, identical


if __name__ == "__main__":
    print("CPX12b: OOF three-gate train-edge binning -- scalar vs batched-across-columns")
    # Realistic FE shapes: many engineered columns, K folds, moderate n.
    bench(n=2000, p=50, n_folds=5)
    bench(n=10000, p=50, n_folds=5)
    bench(n=10000, p=200, n_folds=5)
    bench(n=50000, p=100, n_folds=5)
