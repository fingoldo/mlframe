"""cProfile harness for ``votenrank.adversarial_stochastic_blend``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_adversarial_stochastic_blend``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.adversarial_stochastic_blend import adversarial_stochastic_blend


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _make_dataset(n_samples: int, n_models: int, seed: int):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=n_samples)
    preds = [y_true + 0.3 * rng.standard_normal(n_samples) for _ in range(n_models)]
    test_likeness = rng.uniform(size=n_samples)
    return y_true, preds, test_likeness


def _run(n_samples: int, n_models: int, n_iterations: int) -> None:
    y_true, preds, test_likeness = _make_dataset(n_samples, n_models, seed=0)
    adversarial_stochastic_blend(preds, y_true, test_likeness, _rmse, n_iterations=n_iterations, sample_frac=0.7, n_restarts=2, random_state=0)


def _run_with_diagnostics(n_samples: int, n_models: int, n_iterations: int) -> None:
    """Same workload as ``_run`` but with the opt-in convergence/trustworthiness diagnostics enabled --
    profiled separately to isolate their (post-hoc, vectorized) overhead from the core MC loop above."""
    y_true, preds, test_likeness = _make_dataset(n_samples, n_models, seed=0)
    adversarial_stochastic_blend(
        preds,
        y_true,
        test_likeness,
        _rmse,
        n_iterations=n_iterations,
        sample_frac=0.7,
        n_restarts=2,
        random_state=0,
        track_convergence=True,
        discriminator_auc=0.72,
        fallback_to_uniform_if_untrustworthy=True,
    )


if __name__ == "__main__":
    for n_samples, n_models, n_iterations in [(2000, 3, 50), (2000, 3, 350), (5000, 5, 100)]:
        t0 = time.perf_counter()
        _run(n_samples, n_models, n_iterations)
        wall = time.perf_counter() - t0
        print(f"n_samples={n_samples:>5} n_models={n_models:>2} n_iterations={n_iterations:>4} -> {wall * 1000:9.2f} ms")

    for n_samples, n_models, n_iterations in [(2000, 3, 50), (2000, 3, 350), (5000, 5, 100)]:
        t0 = time.perf_counter()
        _run_with_diagnostics(n_samples, n_models, n_iterations)
        wall = time.perf_counter() - t0
        print(f"[diagnostics] n_samples={n_samples:>5} n_models={n_models:>2} n_iterations={n_iterations:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 3, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_with_diagnostics(2000, 3, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("[diagnostics path]")
    print(buf.getvalue())
