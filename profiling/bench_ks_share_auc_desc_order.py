"""Bench: fast_calibration_report shares the AUC descending argsort with KS.

Before this change fast_calibration_report argsorted the SAME y_pred array twice
per call -- once descending inside fast_aucs_per_group_optimized (for AUC), once
ascending inside ks_statistic. KS now reuses the AUC order (reversed); bit-identical
because the KS kernel folds tied scores into a single CDF jump so within-tie order
of the order array never affects the statistic.

Measured (py3.14 store, NUMBA_DISABLE_CUDA=1, n=100k, warm, best-of-5 x 200 iters):
  CPU-sort path (MLFRAME_METRICS_ARGSORT_GPU_MIN_N=999999999):
    BEFORE 7.9184 ms/call  AFTER 5.5151 ms/call  -> 1.44x, -2.40 ms/call
  default GPU-AUC path:
    BEFORE 8.9932 ms/call  AFTER 6.6018 ms/call  -> 1.36x, -2.39 ms/call
The eliminated KS argsort is always a CPU argsort, so the saving is hardware-independent.

Run:
    PYTHONPATH=src NUMBA_DISABLE_CUDA=1 python profiling/bench_ks_share_auc_desc_order.py
"""
from __future__ import annotations

import time
import numpy as np

import mlframe.metrics.classification._classification_extras as ext
from mlframe.metrics.core import fast_calibration_report, prewarm_numba_cache


def main() -> None:
    prewarm_numba_cache()
    rng = np.random.default_rng(42)
    n = 100_000
    yt = rng.integers(0, 2, n).astype(np.float64)
    yp = np.clip(yt + rng.standard_normal(n) * 0.3, 0.01, 0.99)

    _orig = ext.ks_statistic

    def measure(label: str) -> float:
        for _ in range(3):
            fast_calibration_report(yt, yp, show_plots=False)
        best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(200):
                fast_calibration_report(yt, yp, show_plots=False)
            best = min(best, (time.perf_counter() - t) / 200 * 1000)
        print(f"{label} {best:.4f} ms/call")
        return best

    after = measure("AFTER (shared order)")

    def ks_nosort(yt_, yp_, desc_order=None):
        return _orig(yt_, yp_)

    ext.ks_statistic = ks_nosort
    before = measure("BEFORE (own argsort)")
    print(f"speedup {before / after:.4f}x  saving {before - after:.4f} ms/call")


if __name__ == "__main__":
    main()
