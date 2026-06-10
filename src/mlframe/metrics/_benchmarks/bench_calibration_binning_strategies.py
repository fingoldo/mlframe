"""Profile + wall-time bench for calibration binning at production shape (n=2M).

Compares the three reliability-binning code paths after the INV-3/INV-21 changes:
  - uniform (fast_calibration_binning, now with sum(y_pred) accumulation per pocket)
  - quantile (calibration_binning strategy="quantile": np.quantile edges + njit kernel)
  - auto (calibration_binning strategy="auto": picks quantile for rare-event base rate)

Run::

    python -m mlframe.metrics._benchmarks.bench_calibration_binning_strategies

Conclusion (this host, n=2_000_000, nbins=10, warm numba; 20-call mean):
  - uniform   ~7.7-8.5 ms / call  (single O(n) pass: 3 accumulators incl. pred_sum)
  - quantile  ~79-81 ms / call    (np.quantile O(n log n) partition dominates + 1 njit pass)
  - auto      ~10 ms on balanced (-> uniform), ~77 ms on rare-event (-> quantile)
  cProfile shows the uniform path is entirely inside the compiled numba kernel
  (near-zero Python frames). The INV-3 pred_sum accumulator is one extra float add
  per sample -- no measurable change vs the pre-change ~8 ms; the kernel stays
  njit-fast, NO ACTIONABLE SPEEDUP. Quantile's ~10x cost is the inherent sort for
  equal-population edges; it is selected only for rare-event data where uniform
  collapses to <=2 bins, so the cost is paid solely when it buys a readable diagram
  (and only once per per-class report, not in the ICE / early-stopping hot loop).
"""
from __future__ import annotations

import cProfile
import pstats
import io
from timeit import default_timer as timer

import numpy as np


def _make_data(n: int, rare: bool):
    rng = np.random.default_rng(0)
    if rare:
        p = np.clip(rng.beta(0.5, 120.0, n), 0.0, 1.0)
        p[:2] = 0.99
    else:
        p = rng.random(n)
    y = (rng.random(n) < np.clip(p, 0, 1)).astype(np.int64)
    return y, p


def _time(fn, *args, repeats=20):
    fn(*args)  # warm
    t0 = timer()
    for _ in range(repeats):
        fn(*args)
    return (timer() - t0) / repeats * 1e3  # ms/call


def main(n: int = 2_000_000, nbins: int = 10):
    from mlframe.metrics.calibration._calibration_plot import (
        calibration_binning, fast_calibration_binning,
    )

    print(f"n={n:_}, nbins={nbins}\n")
    for rare in (False, True):
        y, p = _make_data(n, rare)
        label = "rare-event (2% base)" if rare else "balanced"
        t_uniform = _time(lambda: fast_calibration_binning(y, p, nbins))
        t_quantile = _time(lambda: calibration_binning(y, p, nbins, "quantile"))
        t_auto = _time(lambda: calibration_binning(y, p, nbins, "auto"))
        nb_u = len(fast_calibration_binning(y, p, nbins)[2])
        nb_q = len(calibration_binning(y, p, nbins, "quantile")[2])
        print(f"[{label}] base_rate={y.mean():.4f}")
        print(f"  uniform  : {t_uniform:7.3f} ms/call  ({nb_u} non-empty bins)")
        print(f"  quantile : {t_quantile:7.3f} ms/call  ({nb_q} non-empty bins)")
        print(f"  auto     : {t_auto:7.3f} ms/call")
        print()

    print("=== cProfile (uniform, n=2M, 20 calls) ===")
    y, p = _make_data(n, rare=False)
    fast_calibration_binning(y, p, nbins)  # warm
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        fast_calibration_binning(y, p, nbins)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
