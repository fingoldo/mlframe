"""Bench: cluster-aggregate per-method MI stacking.

Hot pattern in ``run_cluster_aggregate_step`` (``_cluster_aggregate.py``): for EACH
candidate combiner method (up to 9) on EACH cluster, the aggregate's MI with the target is
scored via::

    agg_mi = mi(np.column_stack([data, binned.astype(data.dtype)]),
                np.array([data.shape[1]]), target,
                np.concatenate([nbins, [quantization_nbins]]), dtype=dtype)

``np.column_stack([data, binned])` rebuilds a full ``(n, n_features+1)`` copy of the entire
binned matrix every method-iteration, but ``mi`` only ever reads columns ``x`` (the binned col)
and ``y`` (``target``) via ``merge_vars`` -- the whole ``data`` block is copied then discarded.

NEW: stack only the ``target`` columns + the binned column into a compact ``(n, |target|+1)``
matrix and remap ``x``/``y`` indices into it. ``merge_vars`` reads only those columns by value,
so the MI is bit-identical (same per-sample values, same nbins, same merge order).

OLD side = real prior code loaded via ``git show HEAD:...`` is unnecessary here: the OLD path is
just the inline ``np.column_stack([data, binned])`` call, reproduced exactly below.

Run: CUDA_VISIBLE_DEVICES="" python bench_cluster_aggregate_mi_compact_stack.py
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.info_theory import mi


def _make(n: int, n_features: int, nbins_val: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, nbins_val, size=(n, n_features), dtype=np.int32)
    binned = rng.integers(0, nbins_val, size=n).astype(np.int32)
    nbins = np.full(n_features, nbins_val, dtype=np.int64)
    target = np.array([0], dtype=np.int64)  # target is column 0 of data
    return data, binned, nbins, target


def old_score(data, binned, nbins, target, qnb, dtype=np.int32) -> float:
    return float(
        mi(
            np.column_stack([data, binned.astype(data.dtype)]),
            np.array([data.shape[1]], dtype=np.int64),
            target,
            np.concatenate([np.asarray(nbins), [int(qnb)]]).astype(np.int64),
            dtype=dtype,
        )
    )


def new_score(data, binned, nbins, target, qnb, dtype=np.int32) -> float:
    # Compact stack: only the target columns + the binned aggregate. ``mi`` reads
    # only x (binned) and y (target) columns; the rest of ``data`` is never touched.
    tcols = np.asarray(target, dtype=np.int64)
    compact = np.column_stack([data[:, tcols], binned.astype(data.dtype)])
    n_t = tcols.shape[0]
    y_new = np.arange(n_t, dtype=np.int64)
    x_new = np.array([n_t], dtype=np.int64)
    compact_nbins = np.concatenate(
        [np.asarray(nbins)[tcols], [int(qnb)]]
    ).astype(np.int64)
    return float(mi(compact, x_new, y_new, compact_nbins, dtype=dtype))


def bench(n, n_features, nbins_val=10, n_methods=9, iters=30):
    data, binned, nbins, target = _make(n, n_features, nbins_val)
    qnb = nbins_val
    # identity
    o = old_score(data, binned, nbins, target, qnb)
    nw = new_score(data, binned, nbins, target, qnb)
    assert o == nw, f"identity FAIL: {o} vs {nw} (diff {abs(o - nw)})"

    def run(fn):
        best = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter()
            for _m in range(n_methods):  # one cluster scores n_methods candidates
                fn(data, binned, nbins, target, qnb)
            best = min(best, time.perf_counter() - t0)
        return best

    run(old_score); run(new_score)  # warm
    to = run(old_score)
    tn = run(new_score)
    print(f"n={n:>7} feats={n_features:>4} methods={n_methods}: "
          f"OLD {to*1e3:8.3f}ms  NEW {tn*1e3:8.3f}ms  speedup {to/tn:5.2f}x  (mi={o:.6f})")


if __name__ == "__main__":
    for n, f in [(2000, 50), (5000, 200), (20000, 200), (20000, 800), (100000, 200)]:
        bench(n, f)
