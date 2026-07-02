"""cProfile harness for the MoE selection gate (fit + route) at a production-representative shape.

Run: ``CUDA_VISIBLE_DEVICES="" python -m mlframe.training.composite._benchmarks.bench_moe_gate``

Shape: 1M rows / 500 groups / 3 experts (composite, raw, lag). Warms the njit kernel first (cache=True), then
profiles a cold-input fit + a route on the same rows. The hot step is the per-group weighted-SSE reduction
(:func:`_grouped_sse_njit`), a fused single-pass njit kernel; ``pd.factorize`` of the group ids is the next
O(n) cost; routing is a vectorized gather + priority NaN-fallback. See the module docstring for the verdict.
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.training.composite._moe_gate import MoESelectionGate


def _make(n=1_000_000, n_groups=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    g = rng.integers(0, n_groups, n)
    comp = y + rng.normal(0, 0.3, n)
    raw = y + rng.normal(0, 0.5, n)
    lag = y + rng.normal(0, 0.4, n)
    return y, {"composite": comp, "raw": raw, "lag": lag}, g


def main() -> None:
    y, preds, g = _make()
    # Warm the njit kernel (compile is a one-time cost unrelated to steady state).
    MoESelectionGate().fit(y[:1000], {k: v[:1000] for k, v in preds.items()}, group_ids=g[:1000])

    t0 = time.perf_counter()
    gate = MoESelectionGate().fit(y, preds, group_ids=g)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    gate.predict(preds, group_ids=g)
    t_route = time.perf_counter() - t0
    print(f"warm wall: fit {t_fit * 1e3:.1f} ms, route {t_route * 1e3:.1f} ms  (n={y.shape[0]}, groups=500)")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        gate = MoESelectionGate().fit(y, preds, group_ids=g)
        gate.predict(preds, group_ids=g)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(12)


if __name__ == "__main__":
    main()
