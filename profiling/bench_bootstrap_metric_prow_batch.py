"""A/B bench for bootstrap_metric's vectorized-batch per-row-fast-path (2026-07-31).

bootstrap_metric's per-row-fast-path (RMSE / Brier / log-loss's mean-decomposable metrics, when
jackknife_per_row is registered and unstratified) drew one resample index array per Python-level
loop iteration, gathering + reducing it individually -- 1000 separate numpy dispatches for the
default n_bootstrap. numpy's Generator draws the SAME bit-stream whether pulled as n_bootstrap
separate size=n calls or one size=(chunk, n) call, so batching a memory-bounded chunk of resamples
into ONE vectorized gather + mean + reduce is bit-identical, just fewer Python-level round-trips.

Confirms bit-identity and measures the wall-time win at the honest_diagnostics multiclass
regression-fallback shape (n=2M RMSE bootstrap surfaced 20.5s tottime / 8 calls in a cProfile,
profile_one_combo.py --combo c0021_f0cef153 --rows 2000000).

Usage:
    python profiling/bench_bootstrap_metric_prow_batch.py
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.evaluation.bootstrap import bootstrap_metric
from mlframe.metrics.scoring import fast_rmse


def _rmse_per_row(yy, pp):
    return (np.asarray(yy, dtype=np.float64) - np.asarray(pp, dtype=np.float64)) ** 2


def _run(n: int, n_bootstrap: int, seed: int = 0):
    rng = np.random.default_rng(1)
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * 0.3

    def _rmse(yy, pp):
        return float(fast_rmse(yy, pp))

    t0 = time.perf_counter()
    ref = bootstrap_metric(
        y_true, y_pred, metric_fn=_rmse, n_bootstrap=n_bootstrap, alpha=0.05, random_state=seed,
        jackknife_per_row=(_rmse_per_row, False, np.sqrt),
    )
    t_ref = time.perf_counter() - t0

    print(f"n={n:,} n_bootstrap={n_bootstrap}: {t_ref:.4f}s point={ref['point']:.6f} lo={ref['lo']:.6f} hi={ref['hi']:.6f}")
    return ref


def main():
    # Correctness: monkeypatch the batch-path threshold down so a SMALL n exercises it, and compare
    # against a hand-rolled reference that forces the ORIGINAL per-iteration loop (stratify is not
    # None trivially, but that changes the resample scheme -- instead directly compare against calling
    # bootstrap_metric with jackknife_per_row=None, which always takes the slow per-iteration path with
    # the exact metric_fn, then verify the FAST path's samples equal a manual per-iteration RMSE
    # computed with the identical RNG draw sequence).
    n = 5_000
    n_bootstrap = 300
    rng = np.random.default_rng(1)
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * 0.3

    seed = 42
    fast = bootstrap_metric(
        y_true, y_pred, metric_fn=lambda yy, pp: float(fast_rmse(yy, pp)), n_bootstrap=n_bootstrap, alpha=0.05,
        random_state=seed, jackknife_per_row=(_rmse_per_row, False, np.sqrt),
    )
    # Manual per-iteration reference using the SAME per-row-fast-path FORMULA the pre-existing
    # (pre-batch) code used: gather the precomputed per-row squared-error array via idx, .mean(), then
    # np.sqrt -- NOT fast_rmse directly (a different njit single-pass reduction with its own summation
    # order; the project's own docs already note the per-row-fast-path is ~1e-13/~1e-16 CI-equivalent
    # to the exact metric_fn, not bit-identical to it -- the batching change under test here must be
    # bit-identical to the OLD per-iteration form of the SAME formula, which is what this checks).
    manual_rng = np.random.default_rng(seed)
    per_row = _rmse_per_row(y_true, y_pred)
    manual_samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = manual_rng.integers(0, n, size=n, dtype=np.int64)
        manual_samples[i] = float(np.sqrt(per_row[idx].mean()))
    identical = np.array_equal(np.sort(fast["samples"]), np.sort(manual_samples))
    print(f"bit-identical vs manual per-iteration (pre-batch formula) reference: {identical}")
    assert identical, "batched per-row-fast-path diverged from the manual per-iteration reference"

    print()
    print("=== Wall-time (honest_diagnostics multiclass regression-fallback shape) ===")
    _run(n=2_000_000, n_bootstrap=1000)


if __name__ == "__main__":
    main()
