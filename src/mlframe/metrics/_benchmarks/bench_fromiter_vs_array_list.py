"""Microbench: `np.array(list(dict.values()))` vs `np.fromiter(dict.values(), ...)`.

Driven by the user observation 2026-05-20 that the metrics/core.py
per-group outlier-detection block did the list-materialization first
then np.array, when np.fromiter consumes the dict_values view
directly.

Result on the host that produced this bench (cpython 3.11.5,
numpy 2.x, Windows 10, ran 2026-05-20):

    size    list+np.array    np.fromiter    speedup
    ----------------------------------------------------
        10          1.20us        1.19us     1.01x
       100          5.71us        3.33us     1.72x
      1000         50.18us       22.79us     2.20x
     10000        469.72us      220.68us     2.13x
    100000       4802.37us     2165.49us     2.22x

Consistent ~2x speedup at the realistic operating point of
``evaluate_metrics_per_factor`` (n_groups typically 50-2000).
Small (n=10) sees no meaningful difference because Python loop
overhead dominates either approach. Memory footprint is the
additional win at every size: list materialization allocates two
buffers (Python list + ndarray), fromiter just one.
"""
from __future__ import annotations

import gc
import timeit

import numpy as np


def _bench(size: int, n_iter: int = 1000) -> dict:
    """Return {(approach): mean us per call} for the two shapes."""
    rng = np.random.default_rng(42)
    # Build a representative dict the way evaluate_metrics_per_factor would.
    keys = [f"bin_{i} [n=...]" for i in range(size)]
    vals = rng.normal(size=size).tolist()
    d = dict(zip(keys, vals))

    # Old: list + np.array
    def _old():
        return np.array(list(d.values()))

    # New: fromiter
    def _new():
        return np.fromiter(d.values(), dtype=np.float64, count=len(d))

    # Warmup
    for _ in range(3):
        _old()
        _new()
    gc.collect()

    old_t = timeit.timeit(_old, number=n_iter) / n_iter * 1e6
    new_t = timeit.timeit(_new, number=n_iter) / n_iter * 1e6
    return {"old_us": old_t, "new_us": new_t, "speedup": old_t / new_t if new_t > 0 else float("inf")}


def main() -> None:
    print(f"{'size':>8s} {'list+np.array':>16s} {'np.fromiter':>14s} {'speedup':>10s}")
    print("-" * 52)
    for size in (10, 100, 1_000, 10_000, 100_000):
        # Fewer iterations at large sizes to keep total bench time bounded.
        n_iter = 10_000 if size <= 100 else (1000 if size <= 1000 else (100 if size <= 10_000 else 30))
        out = _bench(size, n_iter=n_iter)
        print(f"{size:8d} {out['old_us']:>13.2f}us {out['new_us']:>11.2f}us " f"{out['speedup']:>8.2f}x")


if __name__ == "__main__":
    main()
