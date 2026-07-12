"""cProfile harness for ``votenrank.correlation_diversity_ablation.diversity_ablation_report``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_correlation_diversity_ablation``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.correlation_diversity_ablation import diversity_ablation_report, recommend_diversity_additions


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _make_dataset(n_samples: int, n_models: int, seed: int):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=n_samples)
    oof_preds = {}
    for i in range(n_models):
        noise_scale = 0.2 if i < n_models // 2 else 1.5
        oof_preds[f"m{i}"] = y_true + noise_scale * rng.standard_normal(n_samples)
    individual_scores = {name: -_rmse(y_true, pred) for name, pred in oof_preds.items()}
    return y_true, oof_preds, individual_scores


def _run(n_samples: int, n_models: int, use_greedy_search: bool = False) -> None:
    y_true, oof_preds, individual_scores = _make_dataset(n_samples, n_models, seed=0)
    diversity_ablation_report(oof_preds, individual_scores, y_true, _rmse, use_greedy_search=use_greedy_search)


def _run_recommend(n_samples: int, n_models: int, top_k=None) -> None:
    y_true, oof_preds, individual_scores = _make_dataset(n_samples, n_models, seed=0)
    recommend_diversity_additions(oof_preds, individual_scores, y_true, _rmse, top_k=top_k)


if __name__ == "__main__":
    for use_greedy_search in (False, True):
        label = "greedy_search" if use_greedy_search else "plain"
        for n_samples, n_models in [(2000, 10), (20000, 10), (20000, 50)]:
            t0 = time.perf_counter()
            _run(n_samples, n_models, use_greedy_search=use_greedy_search)
            wall = time.perf_counter() - t0
            print(f"[{label:>13}] n_samples={n_samples:>6} n_models={n_models:>3} -> {wall * 1000:9.2f} ms")

    # The recommender wraps diversity_ablation_report's own O(n_models * n_samples) cost with a cheap
    # filter+sort over the (small, already-flagged) candidate list -- profiled separately to confirm the
    # ranking step itself never becomes the bottleneck as n_models grows.
    for n_samples, n_models in [(2000, 10), (20000, 10), (20000, 50)]:
        t0 = time.perf_counter()
        _run_recommend(n_samples, n_models, top_k=5)
        wall = time.perf_counter() - t0
        print(f"[{'recommend':>13}] n_samples={n_samples:>6} n_models={n_models:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20000, 50, use_greedy_search=False)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("plain path cProfile:")
    print(buf.getvalue())

    # The greedy search's cost is dominated by re-evaluating loss_fn once per (remaining candidate, step) --
    # O(n_flagged^2) calls in the worst case (no early stop) vs the plain path's O(n_flagged) -- profile it
    # separately so its own hotspot (loss_fn re-evaluation, not the pairwise-correlation setup) is visible.
    profiler_greedy = cProfile.Profile()
    profiler_greedy.enable()
    _run(20000, 50, use_greedy_search=True)
    profiler_greedy.disable()
    buf_greedy = StringIO()
    stats_greedy = pstats.Stats(profiler_greedy, stream=buf_greedy).sort_stats("cumulative")
    stats_greedy.print_stats(20)
    print("greedy_search path cProfile:")
    print(buf_greedy.getvalue())

    profiler_recommend = cProfile.Profile()
    profiler_recommend.enable()
    _run_recommend(20000, 50, top_k=5)
    profiler_recommend.disable()
    buf_recommend = StringIO()
    stats_recommend = pstats.Stats(profiler_recommend, stream=buf_recommend).sort_stats("cumulative")
    stats_recommend.print_stats(20)
    print("recommend_diversity_additions path cProfile:")
    print(buf_recommend.getvalue())
