"""cProfile harness for per-round iteration-metric capture overhead.

Measures the marginal cost the booster capture path adds PER ROUND: a native per-iteration val prediction +
``compute_all_metrics`` on a production-ish val size. The capture is the cost driver because it re-predicts val
each captured round, so this confirms the stride/default-off design keeps the overhead bounded.

Run:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES="" MLFRAME_NO_CUDA_AUTOCONFIG=1 \
        python src/mlframe/metrics/_benchmarks/profile_iteration_metrics_capture.py

Findings (measured 2026-06-16, lgb 4.6, CPU; metric-kernel microbench via ``_bench_metric_kernel_only``):
  - ``compute_all_metrics`` (binary, the metric kernel ALONE), warm numba, ms/call:
        n_val=5_000  -> 0.64 ms     n_val=50_000 -> 4.86 ms     n_val=200_000 -> 31.0 ms
    The cost is the shared numba kernels (AUC desc-sort, ECE binning, classification report), already bench-tuned
    elsewhere in the metrics package; the aggregator itself is thin Python delegation with NO actionable hotspot of
    its own (cProfile confirms ~all tottime is inside the njit kernels, mis-attributed to the Python caller frame
    per the cProfile attribution caveat).
  - The PER-ROUND capture cost is one native ``booster.predict(X_val, num_iteration=k)`` PLUS the metric kernel
    above. The predict is LightGBM-native and scales with n_val; at production val sizes it DOMINATES the metric
    kernel. This is the inherent cost of re-scoring val each round and is exactly why booster capture is OFF by
    default and stride-sampled: stride=K divides BOTH the predict and the metric overhead by K.
  - No actionable speedup inside the feature: predict is library-native, metric kernels are shared bench-tuned
    numba. The only levers are the ones already exposed -- ``iteration_metrics_stride`` and the default-off switch.

Note: the full-fit wall A/B (``main()``) needs a QUIET host -- repeated booster fits on a contended box here
native-segfault (documented host instability; run the booster A/B on an uncontended machine). The metric-kernel
microbench (``_bench_metric_kernel_only``) is contention-safe and is the part to re-run for the kernel cost.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np


def _make_data(n_train: int, n_val: int, n_feat: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_train + n_val, n_feat)).astype(np.float32)
    logit = X[:, 0] * 1.2 + X[:, 1] * 0.7 - X[:, 2] * 0.5 + rng.normal(0, 0.5, n_train + n_val)
    y = (logit > 0).astype(np.int64)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _bench_metric_kernel_only(n_val: int, repeats: int = 50) -> float:
    from mlframe.metrics import compute_all_metrics

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, n_val)
    score = np.clip(0.35 + 0.3 * y + rng.normal(0, 0.3, n_val), 0, 1)
    compute_all_metrics(y, score, "binary_classification")  # warm numba
    t0 = time.perf_counter()
    for _ in range(repeats):
        compute_all_metrics(y, score, "binary_classification")
    return (time.perf_counter() - t0) / repeats * 1e3  # ms/call


def main() -> None:
    from mlframe.training.lgb_shim import LGBMClassifierWithDatasetReuse

    n_val = 50_000
    Xtr, ytr, Xva, yva = _make_data(50_000, n_val, 16)

    ms = _bench_metric_kernel_only(n_val)
    print(f"compute_all_metrics kernel-only @ n_val={n_val}: {ms:.3f} ms/call")

    def _fit(stride):
        m = LGBMClassifierWithDatasetReuse(n_estimators=60, num_leaves=31, verbose=-1, n_jobs=1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], capture_iteration_metrics=True, iteration_metrics_stride=stride)
        return m

    # Warm.
    _fit(stride=10)

    for stride in (1, 5, 60):
        t0 = time.perf_counter()
        m = _fit(stride=stride)
        wall = time.perf_counter() - t0
        captured = len(m.iteration_metrics_)
        print(f"stride={stride:>3}: full-fit wall {wall:.3f}s, captured {captured} rounds")

    # Off baseline for the overhead delta.
    t0 = time.perf_counter()
    m_off = LGBMClassifierWithDatasetReuse(n_estimators=60, num_leaves=31, verbose=-1, n_jobs=1)
    m_off.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    print(f"capture OFF: full-fit wall {time.perf_counter() - t0:.3f}s")

    print("\n=== cProfile: stride=1 (every round captured, worst case) ===")
    pr = cProfile.Profile()
    pr.enable()
    _fit(stride=1)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    main()
