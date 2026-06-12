"""Microbench for the two split/fit perf wins.

Run: python -m mlframe.training._benchmarks.bench_split_3way_and_cb_plot

lead1 -- multilabel stratified split: the old path called
MultilabelStratifiedShuffleSplit TWICE (test carve over all N, then val carve
over the remainder), each re-running the full O(n*K*iters) greedy. The new
_stratified_split_3way does ONE greedy IterativeStratification pass into three
folds. Measured n=50k, K=6:  OLD ~1118ms -> NEW ~622ms == 1.80x (~496ms saved),
split valid/disjoint/deterministic, per-label rate deviation <= 6e-5.

lead2 -- CatBoost .fit(plot=...): the wrapper omitted plot=, so CB instantiates
a MetricVisualizer/ipywidget even headless. _maybe_disable_cb_plot injects
plot=False outside an interactive kernel. Pure-config, no numerics change.
Measured cost saved depends on the widget backend (25ms minimum on a headless
dev host; up to ~1.5s/fit where the ipywidgets render thread actually spawns).
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.training._split_helpers import _stratified_split, _stratified_split_3way


def _old_two_call(idx, y, seed):
    tr, te = _stratified_split(idx, 0.2, y, seed)
    strat = y[tr]
    tr2, va = _stratified_split(tr, 0.25, strat, seed)  # val frac of remainder
    return tr2, va, te


def bench_split(n=50_000, k=6, reps=3):
    rng = np.random.default_rng(0)
    y = (rng.random((n, k)) < 0.25).astype(int)
    idx = np.arange(n)

    t = time.perf_counter()
    for _ in range(reps):
        _old_two_call(idx, y, 42)
    old_ms = (time.perf_counter() - t) / reps * 1000

    t = time.perf_counter()
    for _ in range(reps):
        _stratified_split_3way(idx, 0.2, 0.2, y, 42)
    new_ms = (time.perf_counter() - t) / reps * 1000

    print(f"[lead1] n={n} K={k}: OLD {old_ms:.1f}ms -> NEW {new_ms:.1f}ms "
          f"({old_ms / new_ms:.2f}x, {old_ms - new_ms:.0f}ms saved)")


def bench_cb_plot(n=4000, iters=80, reps=3):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("[lead2] catboost not installed; skipping")
        return
    rng = np.random.default_rng(0)
    X = rng.random((n, 12))
    y = (rng.random(n) < 0.5).astype(int)

    def fit(plot):
        m = CatBoostClassifier(iterations=iters, verbose=0, allow_writing_files=False)
        t = time.perf_counter()
        m.fit(X, y, plot=plot)
        return time.perf_counter() - t

    fit(False)  # warm
    tp = min(fit(True) for _ in range(reps)) * 1000
    tf = min(fit(False) for _ in range(reps)) * 1000
    print(f"[lead2] n={n} iters={iters}: plot=True {tp:.0f}ms -> "
          f"plot=False {tf:.0f}ms ({tp - tf:.0f}ms saved)")


if __name__ == "__main__":
    bench_split()
    bench_cb_plot()
