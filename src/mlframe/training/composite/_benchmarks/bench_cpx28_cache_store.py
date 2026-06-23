"""CPX28 bench: DiscoveryCache sequential set/get throughput.

The pre-fix store rewrote the whole ``.lru`` JSON file on every touch and
``glob`` + ``getsize`` over all ``*.pkl`` on every ``set`` (eviction sizing),
giving O(N^2) cost over a run. This bench drives N sequential set + get ops
against a temp dir and reports total wall.

Run (CPU-only, python on PATH):
    CUDA_VISIBLE_DEVICES="" python src/mlframe/training/composite/_benchmarks/bench_cpx28_cache_store.py
"""

from __future__ import annotations

import os
import tempfile
import time


def _run(n: int, max_entries: int) -> float:
    from mlframe.training.composite.cache_store import DiscoveryCache

    with tempfile.TemporaryDirectory() as d:
        cache = DiscoveryCache(d, max_entries=max_entries, max_size_mb=None)
        payload = {"spec": list(range(20)), "meta": 1.0}
        t0 = time.perf_counter()
        for i in range(n):
            key = f"{i:08x}"
            cache.set(key, payload)
            cache.get(key)
        # Flush any pending LRU state (post-fix: close()).
        close = getattr(cache, "close", None)
        if close is not None:
            close()
        return time.perf_counter() - t0


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # Warm (JIT/imports/fs caches).
    _run(50, max_entries=1000)
    for n in (2000, 5000):
        best = min(_run(n, max_entries=1000) for _ in range(3))
        print(f"N={n}: best wall = {best:.4f}s  ({1000 * best / n:.3f} ms/op-pair)")


if __name__ == "__main__":
    main()
