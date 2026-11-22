"""Expanded A/B/C/D(+OOS) validation suite for MDLP validated-splitting (2026-07-19).

Measures whether the significance-gated default (``mdlp_bin_edges(fast_mode=False)``, wired as
production default in ``supervised_binning.py``) generalizes better than the classic depth-capped
path (``fast_mode=True``) across a broad, named scenario matrix -- not just the two synthetic +
two real-column cases from the first-round bench (``bench_mdlp_validated_split_ab.py``, kept
as-is; this module supersedes it as the reference suite going forward).

Two speeds, both runnable from this file:

  * ``run_fast_subset()`` -- ~6 representative scenarios at small n (a few seconds total). This is
    what ``tests/feature_selection/discretization/test_mdlp_validated_split_fast.py`` calls as a
    real pytest -- it hits every code path (noise, signal, NaN, fast_mode parity, OOS variant) at
    a cost cheap enough to run on every test invocation.
  * ``run_full_sweep()`` -- the full scenario x n x distribution matrix (several minutes; NOT run
    by pytest). Run standalone: ``python -m mlframe.feature_selection.filters._benchmarks.bench_mdlp_validated_split_suite --full``

Scope decisions (to keep the full sweep bounded, not a combinatorial explosion):
  * Row counts {1000, 10000, 50000, 200000} are swept ONLY for the two canonical scenarios (pure
    noise, 3-breakpoint step) -- that already answers "does cost/accuracy shift with scale."
    Validated/OOS variants are SKIPPED at n=200000 (only quantile baseline + fast_mode run there):
    an isolated single-column measurement at n=50000 with the permutation-fallback path already
    costs 5-12s; at n=200000 the recursion visits ~4x more nodes at a larger per-node permutation
    cost, previously estimated (not re-measured at 200k here) to add minutes per column -- clearly
    documented as a scope cut, not silently dropped.
  * The distribution x relationship-shape scenarios below are enumerated by NAME (not a full
    cartesian product of every distribution against every relationship) -- each scenario is
    hand-picked to stress a specific failure mode (interaction-only signal, multi-modal target,
    heavy-tailed x, extreme-scale x, partial missingness) rather than combinatorially repeating
    the same story.

Metric: same bin-conditional-mean OOS-MSE/RMSE methodology as the first-round bench (train/test
split, per-bin mean(y) fit on train, scored on held-out test).
"""
from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass

import numpy as np

from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_oos_validated, mdlp_bin_edges_validated
from mlframe.feature_selection.filters._adaptive_nbins import _edges_from_quantiles
from ._mdlp_bench_shared import (  # noqa: F401 -- re-exported; external callers (bench_mdlp_prefilter_hybrid.py, tests) import the scen_* generators from this module's own path
    SCENARIOS,
    _oos_mse,
    _split,
    scen_cauchy_x,
    scen_extreme_scale,
    scen_interaction_only,
    scen_lognormal_x,
    scen_multimodal_target,
    scen_non_monotonic_sine,
    scen_pure_noise,
    scen_step_k_breakpoints,
    scen_with_nan,
)

logger = logging.getLogger(__name__)


@dataclass
class Result:
    scenario: str
    n: int
    method: str
    wall: float
    bins: int
    rmse: float
    note: str = ""


def run_one(scenario: str, n: int, method: str, seed: int = 0) -> Result:
    x, y = SCENARIOS[scenario](n, seed)
    # NaN-bearing x is handled by the train/test split BEFORE calling the binner (matches how a
    # real caller would use these fitters -- edges fit on train, applied to test); the binner's
    # OWN NaN contract (drop NaN rows internally) is exercised separately by the fast pytest.
    x_finite = np.isfinite(x)
    train_idx, test_idx = _split(np.arange(n)[x_finite], y[x_finite], seed=seed)
    x_all, y_all = x[x_finite], y[x_finite]
    x_tr, y_tr = x_all[train_idx], y_all[train_idx]
    x_te, y_te = x_all[test_idx], y_all[test_idx]

    t0 = time.perf_counter()
    if method == "quantile5":
        edges = np.concatenate([[-np.inf], _edges_from_quantiles(x_tr, 5), [np.inf]])
    elif method == "fast_mode":
        edges = mdlp_bin_edges(x_tr, y_tr, fast_mode=True)
    elif method == "validated":
        edges = mdlp_bin_edges(x_tr, y_tr, fast_mode=False)
    elif method == "oos_validated":
        edges = mdlp_bin_edges_oos_validated(x_tr, y_tr)
    else:
        raise ValueError(method)
    wall = time.perf_counter() - t0
    mse, n_bins = _oos_mse(x_tr, y_tr, x_te, y_te, edges)
    rmse = math.sqrt(mse) if np.isfinite(mse) else float("nan")
    return Result(scenario, n, method, wall, n_bins, rmse)


def run_fast_subset() -> list:
    """~6 scenarios at n=3000, all 4 methods -- a few seconds total. Used by the pytest fast test."""
    results = []
    for scenario in ("pure_noise", "step_2bp", "non_monotonic_sine", "interaction_only", "nan_10pct", "extreme_scale_x"):
        for method in ("quantile5", "fast_mode", "validated", "oos_validated"):
            results.append(run_one(scenario, 3000, method))
    return results


def run_full_sweep() -> list:
    results = []
    # Row-count sweep on the two canonical scenarios; validated/oos SKIPPED at n=200000 (see module
    # docstring scope note) -- only the cheap baselines run there to show the cost floor still holds.
    for scenario in ("pure_noise", "step_2bp"):
        for n in (1000, 10000, 50000):
            for method in ("quantile5", "fast_mode", "validated", "oos_validated"):
                results.append(run_one(scenario, n, method))
        for method in ("quantile5", "fast_mode"):
            results.append(run_one(scenario, 200_000, method))
    # Named-scenario matrix at a fixed mid-size n.
    for scenario in (
        "step_5bp", "step_10bp", "non_monotonic_sine", "multimodal_target", "interaction_only",
        "lognormal_x", "cauchy_x", "extreme_scale_x", "nan_1pct", "nan_10pct", "nan_30pct",
    ):
        for method in ("quantile5", "fast_mode", "validated", "oos_validated"):
            results.append(run_one(scenario, 20_000, method))
    return results


def print_report(results: list) -> None:
    print(f"{'scenario':22s} {'n':>7s} {'method':14s} {'wall(ms)':>10s} {'bins':>6s} {'RMSE':>12s}")
    for r in results:
        print(f"{r.scenario:22s} {r.n:7d} {r.method:14s} {r.wall*1000:10.2f} {r.bins:6d} {r.rmse:12.4f}")


# Ground-truth MRMR harness: moved to ``_benchmarks/nbins_strategy/gt_sweep.py`` and re-exported here so existing callers keep working.
from mlframe.feature_selection._benchmarks.nbins_strategy.gt_sweep import (  # noqa: F401
    MRMR_BINNING_METHODS,
    DownstreamQuality,
    MrmrGTResult,
    MulticolumnGT,
    _MRMR_FAST_SUBSET_METHODS,
    _MRMR_HARNESS_FIXED_KWARGS,
    _WELLBORE_DATA_PATH,
    _append_jsonl_result,
    _downstream_quality,
    _prf,
    load_jsonl_results,
    print_mrmr_gt_report,
    run_mrmr_fast_subset,
    run_mrmr_full_sweep,
    run_mrmr_gt_config,
    run_mrmr_gt_wellbore_config,
    run_mrmr_selection,
    scen_multicolumn,
    scen_wellbore100k,
)

# -----------------------------------------------------------------------------
# Hyperparameter sensitivity grid: alpha / n_permutations / max_y_classes.
#
# ``mdlp_bin_edges_validated`` exposes these three tunables directly (production default:
# alpha=0.05, n_permutations=30, max_y_classes=64 -- see ``_mdlp_validated_split.py``). This
# section sweeps each against RMSE / bin count / wall-time, multi-seed per cell, to check whether
# the shipped defaults are actually near a local optimum or whether an easy accuracy/cost win sits
# elsewhere in the grid.
# -----------------------------------------------------------------------------


@dataclass
class HparamResult:
    scenario: str
    n: int
    alpha: float
    n_permutations: int
    max_y_classes: int
    seed: int
    wall: float
    bins: int
    rmse: float


def run_hparam_one(scenario: str, n: int, alpha: float, n_permutations: int, max_y_classes: int, seed: int) -> HparamResult:
    x, y = SCENARIOS[scenario](n, seed)
    x_finite = np.isfinite(x)
    train_idx, test_idx = _split(np.arange(n)[x_finite], y[x_finite], seed=seed)
    x_all, y_all = x[x_finite], y[x_finite]
    x_tr, y_tr = x_all[train_idx], y_all[train_idx]
    x_te, y_te = x_all[test_idx], y_all[test_idx]

    t0 = time.perf_counter()
    edges = mdlp_bin_edges_validated(x_tr, y_tr, alpha=alpha, n_permutations=n_permutations, max_y_classes=max_y_classes, seed=seed)
    wall = time.perf_counter() - t0
    mse, n_bins = _oos_mse(x_tr, y_tr, x_te, y_te, edges)
    rmse = math.sqrt(mse) if np.isfinite(mse) else float("nan")
    return HparamResult(scenario, n, alpha, n_permutations, max_y_classes, seed, wall, n_bins, rmse)


# Kept small deliberately -- n_permutations=1000 alone is already the dominant per-cell cost
# (permutation-fallback path re-runs the full node scan once per draw), so the fast subset uses a
# reduced grid + n rather than the full published sweep, matching the run_fast_subset()/
# run_full_sweep() cost split used everywhere else in this module.
HPARAM_ALPHA_GRID = (0.01, 0.05, 0.1, 0.2)
HPARAM_NPERM_GRID = (50, 200, 1000)
HPARAM_MAXYC_GRID = (16, 64, 256)


def run_hparam_sweep_fast() -> list:
    """2 scenarios (signal + pure-noise) x reduced grid x 3 seeds at n=1500 -- a few seconds under
    normal njit compilation.

    n_permutations endpoints were originally (50, 1000) -- HPARAM_NPERM_GRID's own low/high values --
    but 1000 permutations at the permutation-fallback path (a full node-scan re-run per draw) costs
    ~1.4s/permutation under NUMBA_DISABLE_JIT=1 (pure Python, used by numba-coverage-nightly.yml to
    measure @njit body coverage), i.e. ~1494s for ONE grid cell alone -- the 24 cells this sweep
    hits at the high endpoint would need ~10 hours total, blowing through the workflow's
    --timeout=3600 per-test cap by ~10x and crashing the xdist worker (confirmed live: an
    'AssertionError' crashitem on this exact test in numba-coverage-nightly run 32616328513).
    Lowered to (30, 50): both endpoints verified to still catch a genuine
    under-permutation failure (n_permutations=10 measurably over-splits pure noise at alpha=0.2,
    bins up to 6 -- the empirical p-value resolution is too coarse to reliably reject false splits
    at that draw count) while (30, 50) passes cleanly (bins<=1) across the full scenario x alpha x
    max_y_classes x seed grid, keeping real margin above the confirmed-failing value. Total sweep
    cost under NUMBA_DISABLE_JIT=1 at (30, 50): ~44 min (measured), comfortably under the 60 min
    per-test timeout.
    """
    results = []
    for scenario in ("step_2bp", "pure_noise"):
        for alpha in (0.01, 0.2):  # endpoints only -- the full 4-point grid is a full-sweep concern
            for n_permutations in (30, 50):
                for max_y_classes in (16, 256):
                    for seed in range(3):
                        results.append(run_hparam_one(scenario, 1500, alpha, n_permutations, max_y_classes, seed))
    return results


def run_hparam_sweep_full() -> list:
    """Full alpha x n_permutations x max_y_classes grid, 5 seeds/cell, across a representative
    scenario subset (signal-bearing, pure-noise, and heavy-tailed x) at n=8000 -- several minutes."""
    results = []
    for scenario in ("step_2bp", "pure_noise", "cauchy_x"):
        for alpha in HPARAM_ALPHA_GRID:
            for n_permutations in HPARAM_NPERM_GRID:
                for max_y_classes in HPARAM_MAXYC_GRID:
                    for seed in range(5):
                        results.append(run_hparam_one(scenario, 8000, alpha, n_permutations, max_y_classes, seed))
    return results


def print_hparam_report(results: list) -> None:
    from collections import defaultdict

    by_key: dict = defaultdict(list)
    for r in results:
        by_key[(r.scenario, r.alpha, r.n_permutations, r.max_y_classes)].append(r)
    print(f"{'scenario':12s} {'alpha':>6s} {'nperm':>6s} {'maxyc':>6s} {'n':>4s} {'wall(ms)':>10s} {'bins':>14s} {'RMSE':>16s}")
    for (scenario, alpha, n_permutations, max_y_classes), rows in by_key.items():
        walls = np.array([r.wall for r in rows])
        bins = np.array([r.bins for r in rows])
        rmses = np.array([r.rmse for r in rows if np.isfinite(r.rmse)])
        rmse_str = f"{rmses.mean():7.3f}+/-{rmses.std():<6.3f}" if rmses.size else "nan"
        print(
            f"{scenario:12s} {alpha:6.2f} {n_permutations:6d} {max_y_classes:6d} {len(rows):4d} "
            f"{walls.mean()*1000:10.2f} {bins.mean():6.2f}+/-{bins.std():<6.2f} {rmse_str:>16s}"
        )


# the duplicate-row/outlier robustness sweep was carved out into
# bench_mdlp_robustness.py to clear the repo's enforced hard 1000-LOC CI gate (this file was 1006
# lines). Re-exported here so the __main__ block below (and external callers, e.g. tests) keep
# working unchanged. Safe as a top-level import (unlike before the SCENARIOS/_oos_mse/_split move
# above into _mdlp_bench_shared.py): bench_mdlp_robustness.py no longer imports anything from
# THIS module, so this is a one-way dependency, not a cycle (test_no_import_cycles).
from .bench_mdlp_robustness import (  # noqa: F401
    DUP_RATES, OUTLIER_KINDS, RobustnessResult,
    _inject_duplicates, _inject_outliers,
    print_robustness_report, run_robustness_fast, run_robustness_full, run_robustness_one,
)


def _main() -> None:
    """Script entry point for ``python -m ...bench_mdlp_validated_split_suite --full``."""
    if "--full" in sys.argv:
        print_report(run_full_sweep())
        print_mrmr_gt_report(run_mrmr_full_sweep())
        print_hparam_report(run_hparam_sweep_full())
        print_robustness_report(run_robustness_full())
    else:
        print_report(run_fast_subset())
        print_mrmr_gt_report(run_mrmr_fast_subset())
        print_hparam_report(run_hparam_sweep_fast())
        print_robustness_report(run_robustness_fast())


if __name__ == "__main__":
    _main()
