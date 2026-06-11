"""cProfile harness for the multilabel THRESHOLD_SWEEP panel at production shape.

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_threshold_sweep``

The per-label F1 sweep computes the (K, T) F1 matrix in one fused njit-parallel pass per label: for each
probability it derives the grid-fire index (count of grid thresholds ``t<=p`` on the uniform grid) inline
and accumulates the positive/negative histograms, then reverse-cumsums to the decreasing TP(t)/FP(t) step
functions -- no per-threshold recompute, no per-label argsort, no length-n temporaries.

Optimization trail:
* v1 pure-numpy ``np.searchsorted`` + ``np.bincount`` per label: ~704 ms at n=1e6/K=10; cProfile showed
  ``searchsorted`` (O(N log T) binary search) was 589 ms of it.
* v2 closed-form uniform-grid fire index in numpy (floor + one comparison correction, bit-identical to
  searchsorted): the correction's fancy-index gathers over n became the new bottleneck (~578 ms).
* v3 (current) fused njit-parallel kernel over the K independent labels: ~22 ms at n=1e6/K=10 (~31x over
  v1), ~115 ms at n=5e6, ~172 ms at n=1e6/K=50. Bit-identical to sklearn f1_score across the FULL 200-pt
  grid and to the numpy fallback (asserted in tests). The fire-index is the exact corrected closed form,
  so which threshold is reported F1-optimal is unchanged.

Conclusion (numbers below): the njit kernel is the floor here -- O(K*N) with one fused typed pass per
label, parallel across labels, no temporaries. No further actionable speedup at one-panel-per-report.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.multilabel import _SWEEP_N_THRESHOLDS, _per_label_f1_sweep


def _make_data(n: int, K: int):
    rng = np.random.default_rng(0)
    y = (rng.random((n, K)) < np.linspace(0.15, 0.55, K)).astype(np.int8)
    proba = np.clip(y * 0.4 + rng.random((n, K)) * 0.5, 0.0, 1.0)
    thresholds = np.linspace(0.0, 1.0, _SWEEP_N_THRESHOLDS)
    return y, proba, thresholds


def _walltime(y, p, t, repeats: int = 3) -> float:
    _per_label_f1_sweep(y, p, t)  # warm
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        _per_label_f1_sweep(y, p, t)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    for n, K in ((100_000, 10), (1_000_000, 10), (5_000_000, 10), (1_000_000, 50)):
        y, p, t = _make_data(n, K)
        ms = _walltime(y, p, t) * 1e3
        print(f"n={n:>9} K={K:>3}  threshold_sweep={ms:8.2f} ms")

    y, p, t = _make_data(1_000_000, 10)
    pr = cProfile.Profile()
    pr.enable()
    _per_label_f1_sweep(y, p, t)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
