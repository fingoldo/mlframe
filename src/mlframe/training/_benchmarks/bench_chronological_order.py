"""Cost of ``chronological_order_index``: the already-sorted fast path (one O(n) check) vs a real reorder (gather + stable argsort).

Run: python -m mlframe.training._benchmarks.bench_chronological_order
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.training._chronological_order import chronological_order_index


def _best(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts), float(np.median(ts))


def main() -> None:
    rng = np.random.default_rng(0)
    for n in (1_000_000, 10_000_000):
        base = np.datetime64("2020-01-01", "ns")
        ts_sorted = base + np.arange(n).astype("timedelta64[s]").astype("timedelta64[ns]")
        ts_shuf = ts_sorted[rng.permutation(n)]
        idx = np.arange(int(n * 0.8))  # train = 80% of rows, source-row order
        reps = 5 if n <= 1_000_000 else 3
        a = _best(lambda: chronological_order_index(idx, ts_sorted), reps)
        b = _best(lambda: chronological_order_index(idx, ts_shuf), reps)
        gather = _best(lambda: ts_shuf.view(np.int64)[idx], reps)
        print(f"n={n:>10,}  already-sorted check: best {a[0]*1e3:8.1f} ms  median {a[1]*1e3:8.1f} ms | "
              f"reorder (shuffled): best {b[0]*1e3:8.1f} ms  median {b[1]*1e3:8.1f} ms | bare gather {gather[0]*1e3:.1f} ms")


if __name__ == "__main__":
    main()
