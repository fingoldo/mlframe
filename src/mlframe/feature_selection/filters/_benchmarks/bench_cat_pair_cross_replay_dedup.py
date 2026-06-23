"""Bench: apply_cat_pair_cross replay -- per-row tuple dict.get loop variants.

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters._benchmarks.bench_cat_pair_cross_replay_dedup

Context
-------
``apply_cat_pair_cross`` (replay) currently does:
    np.array([_value_for_pair(cats_i[r], cats_j[r]) for r in range(n)], ...)
i.e. an INDEXED per-row loop (``cats_i[r]`` / ``cats_j[r]`` are numpy
__getitem__ calls, each boxing a Python object). A prior 2026-06-02 attempt at
a factorize-fold DEDUP was 0.9x (3 hashing passes). This bench instead tests a
pure micro-rewrite of the SAME per-row loop that removes the two per-row numpy
__getitem__ boxings by iterating with ``zip(cats_i.tolist(), cats_j.tolist())``
-- ``.tolist()`` boxes the whole object array to Python objects in ONE C pass,
then ``zip`` yields already-boxed Python objects (no per-row numpy indexing).
Bit-identical by construction (same values, same order, same dict.get).
"""
from __future__ import annotations

import time

import numpy as np


def _make_mapping(ki: int, kj: int, seen_frac: float = 0.8) -> dict:
    rng = np.random.default_rng(0)
    all_pairs = [(f"a{i}", f"b{j}") for i in range(ki) for j in range(kj)]
    n_seen = max(1, int(len(all_pairs) * seen_frac))
    idx = rng.permutation(len(all_pairs))[:n_seen]
    return {all_pairs[k]: c for c, k in enumerate(sorted(idx))}


def old_replay(cats_i, cats_j, mapping, sentinel):
    n = len(cats_i)
    return np.array(
        [float(mapping.get((cats_i[r], cats_j[r]), sentinel)) for r in range(n)],
        dtype=np.float64,
    )


def new_replay(cats_i, cats_j, mapping, sentinel):
    # ``.tolist()`` boxes each object array to a Python list in ONE C pass; the
    # zip then yields pre-boxed Python objects -- no per-row numpy __getitem__.
    get = mapping.get
    return np.array(
        [float(get((si, sj), sentinel)) for si, sj in zip(cats_i.tolist(), cats_j.tolist())],
        dtype=np.float64,
    )


def bench(n: int, ki: int, kj: int, reps: int = 9):
    rng = np.random.default_rng(42)
    cats_i = np.array([f"a{v}" for v in rng.integers(0, ki, size=n)], dtype=object)
    cats_j = np.array([f"b{v}" for v in rng.integers(0, kj, size=n)], dtype=object)
    mapping = _make_mapping(ki, kj)
    sentinel = len(mapping)

    out_old = old_replay(cats_i, cats_j, mapping, sentinel)
    out_new = new_replay(cats_i, cats_j, mapping, sentinel)
    identical = np.array_equal(out_old, out_new)

    def timed(fn):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(cats_i, cats_j, mapping, sentinel)
            best = min(best, time.perf_counter() - t0)
        return best

    t_old = timed(old_replay)
    t_new = timed(new_replay)
    print(
        f"n={n:>8} ki={ki} kj={kj}  OLD={t_old*1e3:8.2f}ms  NEW={t_new*1e3:8.2f}ms  "
        f"speedup={t_old/t_new:5.2f}x  identical={identical}"
    )


if __name__ == "__main__":
    for n in (5_000, 50_000, 200_000):
        bench(n, 15, 15)
    bench(200_000, 50, 50)
    bench(1_000_000, 15, 15)
