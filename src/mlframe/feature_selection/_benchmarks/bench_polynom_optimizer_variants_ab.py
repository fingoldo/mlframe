"""Budget-matched quality A/B: fe_optimizer 'cma_batch' (current default) vs 'numba_kernel' (njit).

Hard synthetic pairs where trivial unary/binary features cannot capture the signal (the exact regime
the polynomial optimiser exists for), multi-seed. Metric: achieved raw MI of the best engineered
feature per (case, seed). Parity bar (accurate-default-first): numba_kernel median MI >= cma_batch
median MI - 2% slack per case.

MEASURED (2026-07-15, 22-core dev box, budget 5 seeds x 5 restarts x 100 trials, n=20k):
  cma_batch    wall=246.9s cpu=89%  cubic_inner=0.4813  cheb_mix=0.3913   <- current default, confirmed best
  random_batch wall=260.0s cpu=85%  cubic_inner=0.4813  cheb_mix=0.3920
  cma          wall=300.8s cpu=91%  cubic_inner=0.4813  cheb_mix=0.3913
  optuna       wall=402.4s cpu=91%  cubic_inner=0.4813  cheb_mix=0.3913   <- 1.6x slower, same quality
  numba_kernel wall=353.9s cpu=67%  cubic_inner=0.0000  cheb_mix=0.3913   <- QUALITY FAIL + slower; default flip REJECTED

numba_kernel returns None on the cubic-inner case (its best never clears the 1.01x baseline-uplift
gate) while every other variant recovers mi=0.4813 from the same warm seeds -- reproducer in
_numba_polynom_optimizer.py's module docstring. bilinear_xor is non-discriminative as constructed
(trivial baseline captures it; all variants 0.0).
"""
import time

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

N = 20000
CASES = {}
_r = np.random.default_rng(7)
a = _r.standard_normal(N); b = _r.standard_normal(N)
CASES["cubic_inner"] = (a, b, ((a**3 - 2 * a) * b > 0).astype(np.int64))          # non-monotone inner
CASES["bilinear_xor"] = (a, b, ((a * b) > 0).astype(np.int64))                    # sign-product
c = _r.standard_normal(N); d = _r.standard_normal(N)
CASES["cheb_mix"] = (c, d, ((2 * c**2 - 1) + (4 * d**3 - 3 * d) + 0.3 * _r.standard_normal(N) > 0).astype(np.int64))

BUDGET = dict(n_trials=100, min_degree=3, max_degree=6, coef_range=(-2.0, 2.0), l2_penalty=0.05,
              sweep_degrees=True, basis="chebyshev", mi_estimator="plugin", discrete_target=True,
              warm_start=True, multi_fidelity=False)
SEEDS = [42, 43, 44, 45, 46]
RESTARTS = 5

import threading

import psutil


class _CpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples: list = []
        self._stop = threading.Event()

    def run(self):
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=None))
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)
        return (float(np.mean(self.samples)), float(np.max(self.samples))) if self.samples else (0.0, 0.0)


if __name__ == "__main__":
    VARIANTS = ("cma_batch", "numba_kernel", "random_batch", "cma", "optuna")
    results = {}
    for opt in VARIANTS:
        per_case = {}
        sampler = _CpuSampler(); sampler.start()
        t0 = time.perf_counter()
        failed = None
        for name, (xa, xb, y) in CASES.items():
            best_mis = []
            for seed in SEEDS:
                best = None
                for ro in range(RESTARTS):
                    try:
                        res = optimise_hermite_pair(x_a=xa, x_b=xb, y=y, seed=seed + ro, optimizer=opt, **BUDGET)
                    except Exception as e:
                        failed = f"{type(e).__name__}: {e}"
                        break
                    if res is not None and (best is None or res.mi > best):
                        best = res.mi
                if failed:
                    break
                best_mis.append(best if best is not None else 0.0)
            if failed:
                break
            per_case[name] = (float(np.median(best_mis)), float(np.min(best_mis)), float(np.max(best_mis)))
        wall = time.perf_counter() - t0
        cpu_avg, cpu_max = sampler.stop()
        if failed:
            print(f"=== {opt}: FAILED ({failed})")
            continue
        results[opt] = (wall, cpu_avg, per_case)
        print(f"=== {opt}: wall={wall:.1f}s cpu_avg={cpu_avg:.0f}% cpu_max={cpu_max:.0f}%")
        for name, (med, lo, hi) in per_case.items():
            print(f"  {name:14s} median={med:.4f} min={lo:.4f} max={hi:.4f}")
