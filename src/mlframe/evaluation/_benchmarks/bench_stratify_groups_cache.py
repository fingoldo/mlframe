"""Bench for PERF #4: caching the per-class stratify groups dict across repeated bootstrap_metric calls.

The groups dict ``{int(c): np.flatnonzero(stratify == c) for c in np.unique(stratify)}`` is deterministic in
``stratify`` and rebuilt on every call. Honest-diagnostics bootstraps many metrics over the SAME stratify vector
(one per-target seed, ~12 metrics), so the rebuild repeats. This measures (a) the isolated rebuild cost vs a full
bootstrap_metric call, and (b) end-to-end repeated-call wall with vs without a stratify-keyed cache.

Run: python -m mlframe.evaluation._benchmarks.bench_stratify_groups_cache
"""
from __future__ import annotations

import time
import numpy as np

from mlframe.evaluation.bootstrap import bootstrap_metric


def _rebuild(stratify):
    return {int(c): np.flatnonzero(stratify == c) for c in np.unique(stratify)}


def bench(n=100_000, n_bootstrap=200, n_calls=12, seed=0):
    rng = np.random.default_rng(seed)
    strat = (rng.random(n) < 0.1).astype(np.int64)
    y_true = strat.copy()
    y_score = rng.random(n)

    # isolated rebuild cost
    t = time.perf_counter()
    for _ in range(50):
        _rebuild(strat)
    rebuild_ms = (time.perf_counter() - t) / 50 * 1e3

    def metric(a, b):
        # cheap metric so the groups rebuild is a visible fraction
        return float(np.mean(a) - np.mean(b))

    # warm
    bootstrap_metric(y_true, y_score, metric, n_bootstrap=n_bootstrap, stratify=strat, random_state=1, method="percentile")

    t = time.perf_counter()
    for c in range(n_calls):
        bootstrap_metric(y_true, y_score, metric, n_bootstrap=n_bootstrap, stratify=strat, random_state=c, method="percentile")
    full_ms = (time.perf_counter() - t) / n_calls * 1e3

    print(f"n={n} n_bootstrap={n_bootstrap} n_calls={n_calls}")
    print(f"  isolated groups rebuild : {rebuild_ms:.3f} ms")
    print(f"  full bootstrap_metric   : {full_ms:.3f} ms/call")
    print(f"  rebuild as % of call    : {rebuild_ms / full_ms * 100:.2f}%")


if __name__ == "__main__":
    for nb in (200, 1000):
        bench(n=100_000, n_bootstrap=nb)
        bench(n=1_000_000, n_bootstrap=nb)
        print()
