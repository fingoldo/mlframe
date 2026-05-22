"""Benchmark four polynom-FE inner-search optimizers on the same problems.

Compares wall-clock and best-MI achieved across:
  1. ``optuna``       -- current default; per-trial Python TPE sampler
                         decision; holds GIL between numba kernel calls.
  2. ``cma``          -- existing CMA-ES; popsize=20 ask/tell, but inner
                         eval is per-solution sequential (one numba MI
                         call per candidate).
  3. ``cma_batch``    -- NEW: same CMA ask/tell loop, but per-generation
                         eval is ONE batched MI call across all popsize
                         (cand, bf) columns.
  4. ``random_batch`` -- NEW: pure batch random + elitism + Gaussian
                         perturbation of current best. No Optuna, no
                         CMA. One MI batch call per iter.

Run::

    python -m mlframe.profiling.bench_polynom_optimizers

Each optimizer gets identical budget (``n_trials``) and the same
warm-start seeds. Wall-clock is the only timing reported; per-trial /
per-generation breakdowns can be derived by dividing by n_trials /
popsize respectively. Best MI is the highest raw MI achieved across
the full search (independent of any l2 penalty applied during
selection).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from mlframe.feature_selection.filters._hermite_fe_optimise_pair import optimise_hermite_pair


N_ROWS = 4000          # smaller than prod 4M for tractability; larger
                       # than the existing perf-regressions tests (1k)
                       # so numba parallel actually fires.
N_PAIRS = 12           # 12 pairs is the median prod count.
N_SEEDS = 3            # repeat each (pair, optimizer) with N seeds.
N_TRIALS = 200         # matches user's fe_smart_polynom_optimization_steps.
MAX_DEGREE = 6         # matches user's fe_max_polynom_degree.
MIN_DEGREE = 3         # matches user's fe_min_polynom_degree.
COEF_RANGE = (-2.0, 2.0)
DISCRETE_TARGET = True   # plugin MI on classified target (matches MRMR).
BASIS = "hermite"
MI_ESTIMATOR = "plugin"


def _make_pair(seed: int):
    """Build (x_a, x_b, y) for one pair with known nonlinear MI structure.

    Target follows ``sign(2*x_a^2 - x_b)``. Captures a quadratic
    interaction the polynomial-pair FE should be able to recover.
    """
    rng = np.random.default_rng(seed)
    x_a = rng.uniform(-1.5, 1.5, N_ROWS).astype(np.float64)
    x_b = rng.uniform(-1.5, 1.5, N_ROWS).astype(np.float64)
    noise = rng.normal(0.0, 0.15, N_ROWS)
    y_continuous = 2.0 * x_a * x_a - x_b + noise
    if DISCRETE_TARGET:
        # Threshold at the median so train classes are ~50/50.
        y = (y_continuous > np.median(y_continuous)).astype(np.int64)
    else:
        y = y_continuous.astype(np.float64)
    return x_a, x_b, y


def _bench_one(optimizer: str, pair_seed: int, opt_seed: int):
    """Run one optimizer on one synthetic pair, return (elapsed_s, best_mi).
    """
    x_a, x_b, y = _make_pair(pair_seed)

    t0 = time.perf_counter()
    res = optimise_hermite_pair(
        x_a=x_a, x_b=x_b, y=y,
        discrete_target=DISCRETE_TARGET,
        max_degree=MAX_DEGREE,
        min_degree=MIN_DEGREE,
        n_trials=N_TRIALS,
        coef_range=COEF_RANGE,
        seed=opt_seed,
        sweep_degrees=True,
        basis=BASIS,
        mi_estimator=MI_ESTIMATOR,
        optimizer=optimizer,
        # Disable multi-fidelity so the batch variants take the same
        # number of evaluations as cma / optuna (multi-fidelity uses
        # eval_pair_fn closures incompatible with batch eval).
        multi_fidelity=False,
    )
    elapsed = time.perf_counter() - t0
    best_mi = float(res.mi) if (res is not None and hasattr(res, "mi")) else float("nan")
    return elapsed, best_mi


def main():
    OPTIMIZERS = ("optuna", "cma", "cma_batch", "random_batch")
    pair_seeds = list(range(N_PAIRS))
    opt_seeds = list(range(N_SEEDS))

    # Warmup: trigger numba JIT on a tiny first call so the bench timing
    # doesn't include compile cost in the first cell.
    print("# Warmup (one tiny call per optimizer to JIT all kernels)...")
    for opt in OPTIMIZERS:
        try:
            _bench_one(opt, pair_seed=999, opt_seed=999)
        except Exception as e:
            print(f"# warmup FAILED for {opt}: {e}")
    print()

    print(f"# bench_polynom_optimizers")
    print(f"#   N_ROWS={N_ROWS}, N_PAIRS={N_PAIRS}, N_SEEDS={N_SEEDS}, "
          f"N_TRIALS={N_TRIALS}, degrees={MIN_DEGREE}-{MAX_DEGREE}, "
          f"basis={BASIS}, mi_estimator={MI_ESTIMATOR}")
    print(f"#   target: discrete (sign(2*x_a^2 - x_b + noise) > median)")
    print()

    # results[opt] = list of (elapsed, mi) over all (pair, seed) cells
    results: dict[str, list[tuple[float, float]]] = {opt: [] for opt in OPTIMIZERS}

    for pair_idx, ps in enumerate(pair_seeds):
        for opt in OPTIMIZERS:
            cell_times = []
            cell_mis = []
            for os_ in opt_seeds:
                try:
                    t, mi = _bench_one(opt, pair_seed=ps, opt_seed=os_)
                except Exception as e:
                    print(f"# {opt} pair={pair_idx} seed={os_} FAILED: "
                          f"{type(e).__name__}: {e}")
                    t, mi = float("nan"), float("nan")
                cell_times.append(t)
                cell_mis.append(mi)
                results[opt].append((t, mi))
            t_mean = float(np.nanmean(cell_times))
            mi_mean = float(np.nanmean(cell_mis))
            print(f"  pair={pair_idx:>2}  {opt:<14}  time_mean={t_mean:>7.2f}s "
                  f"mi_mean={mi_mean:>7.4f}  (n_seeds={len(opt_seeds)})")
        print()

    print()
    print("# AGGREGATE")
    print(f"  {'optimizer':<14}  {'time_total':>12}  {'time_mean':>10}  "
          f"{'mi_mean':>10}  {'mi_std':>10}  {'speedup_vs_optuna':>18}")
    base_t = float(np.nanmean([t for t, _mi in results["optuna"]]))
    for opt in OPTIMIZERS:
        ts = np.array([t for t, _mi in results[opt]])
        mis = np.array([mi for _t, mi in results[opt]])
        t_total = float(np.nansum(ts))
        t_mean = float(np.nanmean(ts))
        mi_mean = float(np.nanmean(mis))
        mi_std = float(np.nanstd(mis))
        speedup = base_t / t_mean if t_mean > 0 else float("nan")
        print(f"  {opt:<14}  {t_total:>10.2f}s  {t_mean:>9.3f}s  "
              f"{mi_mean:>10.4f}  {mi_std:>10.4f}  {speedup:>18.2f}x")
    print()

    # MI-quality table: how often does each optimizer find within X% of
    # the best MI for that pair across all four optimizers?
    print("# MI-quality: fraction of (pair, seed) cells where optimizer's "
          "MI is within 1% / 5% / 10% of the per-cell best across all "
          "4 optimizers")
    n_cells = len(pair_seeds) * len(opt_seeds)
    # Reshape into (n_cells, optimizers).
    matrix = np.full((n_cells, len(OPTIMIZERS)), np.nan, dtype=np.float64)
    for c, opt in enumerate(OPTIMIZERS):
        for r, (_t, mi) in enumerate(results[opt]):
            matrix[r, c] = mi
    per_cell_best = np.nanmax(matrix, axis=1)
    for opt_idx, opt in enumerate(OPTIMIZERS):
        col = matrix[:, opt_idx]
        within = lambda pct: float(np.nanmean(
            col >= per_cell_best * (1.0 - pct / 100.0)
        ))
        print(f"  {opt:<14}  within_1%={within(1):.2f}  "
              f"within_5%={within(5):.2f}  within_10%={within(10):.2f}")
    print()


if __name__ == "__main__":
    main()
