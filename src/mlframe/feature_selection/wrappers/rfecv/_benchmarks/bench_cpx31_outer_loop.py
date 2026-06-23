"""CPX31 bench: the per-outer-iter ``fi_run_order`` keys-rebuild on the growing ``feature_importances`` dict.

The outer while-loop of ``RFECV.fit`` (``_fit_outer_loop.run_outer_loop_iteration``) passed ``fi_run_order=list(state.feature_importances.keys())`` to ``get_next_features_subset`` on EVERY iteration. ``feature_importances`` grows by ~n_splits keys per iteration, so rebuilding the full key list each iter is O(steps^2) over a run. The list is only consumed by ``get_next_features_subset`` when ``fi_decay_rate>0`` (age-weighted voting); the default ``fi_decay_rate=0.0`` never reads it.

FIX: build the list only when ``fi_decay_rate>0``; pass ``None`` otherwise (the consumer already guards on ``None``).

Run: ``CUDA_VISIBLE_DEVICES="" python bench_cpx31_outer_loop.py``. Warm, best-of-N.
"""
from __future__ import annotations

import time


def _bench(n_iters: int, keys_per_iter: int, build_list: bool, repeats: int = 7) -> float:
    """Simulate the keys-rebuild pattern: a dict grows by ``keys_per_iter`` entries each iter; OLD builds list(keys()) every iter, NEW skips it (decay off)."""
    best = float("inf")
    for _ in range(repeats):
        fi: dict = {}
        t0 = time.perf_counter()
        for it in range(n_iters):
            for f in range(keys_per_iter):
                fi[f"{it}_{f}"] = {}
            if build_list:
                _run_order = list(fi.keys())  # OLD: O(len(fi)) each iter -> O(steps^2)
            else:
                _run_order = None  # NEW (decay off): O(1)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def main() -> None:
    print(f"{'n_iters':>8} {'keys/it':>8} | {'OLD list(keys) ms':>18} {'NEW None ms':>14} {'speedup':>9}")
    for n_iters in (500, 1000, 2000):
        keys_per_iter = 5  # ~n_splits FI runs added per outer iter
        old = _bench(n_iters, keys_per_iter, build_list=True)
        new = _bench(n_iters, keys_per_iter, build_list=False)
        print(f"{n_iters:>8} {keys_per_iter:>8} | {old*1e3:>18.3f} {new*1e3:>14.3f} {old/new:>8.1f}x")


if __name__ == "__main__":
    main()
