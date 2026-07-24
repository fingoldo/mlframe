"""Duplicate-row and outlier/heavy-tail robustness sweep, carved out of
``bench_mdlp_validated_split_suite.py`` (X_EFFICIENCY_ARCHITECTURE-1 fix, mrmr_audit_2026-07-22) to
clear the repo's enforced hard 1000-LOC CI gate (that file was 1006 lines). Behaviour preserved
bit-for-bit; the parent re-exports every name so its own ``__main__`` block keeps working unchanged.

Exact duplicate rows inflate the raw row count the significance test sees without adding
independent evidence; check whether this causes over-splitting relative to the same rows
without duplication (matched information content). Outlier contamination is checked separately
for a degenerate/wrong-result failure mode. Compares validated vs classic fast_mode vs the
quantile baseline, multi-seed, distributions reported (not just means).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
from mlframe.feature_selection.filters._adaptive_nbins import _edges_from_quantiles

from .bench_mdlp_validated_split_suite import SCENARIOS, _oos_mse, _split


@dataclass
class RobustnessResult:
    scenario: str
    perturbation: str  # e.g. "dup_0.50" or "outlier_cauchy"
    method: str
    seed: int
    wall: float
    bins: int
    rmse: float


def _inject_duplicates(x: np.ndarray, y: np.ndarray, dup_rate: float, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """Append ``round(dup_rate * n)`` extra rows, each an exact copy of a randomly chosen existing
    row -- inflates apparent n without adding independent (x, y) evidence."""
    rng = np.random.default_rng(seed + 9973)
    n = x.shape[0]
    n_dup = int(round(dup_rate * n))
    if n_dup <= 0:
        return x, y
    dup_idx = rng.integers(0, n, n_dup)
    return np.concatenate([x, x[dup_idx]]), np.concatenate([y, y[dup_idx]])


def _inject_outliers(x: np.ndarray, seed: int, kind: str) -> np.ndarray:
    """``kind='scale'``: 1% of rows multiplied by 100-1000x. ``kind='cauchy'``: 5% of rows replaced
    by heavy-tailed Cauchy-distributed contamination mixed into the normal base."""
    rng = np.random.default_rng(seed + 7331)
    x = x.copy()
    n = x.shape[0]
    if kind == "scale":
        mask = rng.random(n) < 0.01
        n_mask = int(np.count_nonzero(mask))
        factors = rng.uniform(100.0, 1000.0, n_mask) * rng.choice([-1.0, 1.0], n_mask)
        x[mask] = x[mask] * factors
    elif kind == "cauchy":
        mask = rng.random(n) < 0.05
        n_mask = int(np.count_nonzero(mask))
        x[mask] = rng.standard_cauchy(n_mask) * 50.0
    else:
        raise ValueError(kind)
    return x


def run_robustness_one(scenario: str, n: int, perturbation: str, method: str, seed: int) -> RobustnessResult:
    x, y = SCENARIOS[scenario](n, seed)
    if perturbation.startswith("dup_"):
        dup_rate = float(perturbation.split("_", 1)[1])
        x, y = _inject_duplicates(x, y, dup_rate, seed)
    elif perturbation.startswith("outlier_"):
        x = _inject_outliers(x, seed, perturbation.split("_", 1)[1])
    elif perturbation != "baseline":
        raise ValueError(perturbation)

    x_finite = np.isfinite(x)
    n_eff = x.shape[0]
    train_idx, test_idx = _split(np.arange(n_eff)[x_finite], y[x_finite], seed=seed)
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
    else:
        raise ValueError(method)
    wall = time.perf_counter() - t0
    mse, n_bins = _oos_mse(x_tr, y_tr, x_te, y_te, edges)
    rmse = math.sqrt(mse) if np.isfinite(mse) else float("nan")
    return RobustnessResult(scenario, perturbation, method, seed, wall, n_bins, rmse)


DUP_RATES = (0.0, 0.10, 0.50, 0.90)
OUTLIER_KINDS = ("scale", "cauchy")


def run_robustness_fast() -> list:
    """1 noise scenario + 1 signal scenario x dup-rate grid x 3 methods x 3 seeds, plus outlier
    contamination on the signal scenario -- a few seconds total."""
    results = []
    for scenario in ("pure_noise", "step_2bp"):
        for dup_rate in DUP_RATES:
            for method in ("quantile5", "fast_mode", "validated"):
                for seed in range(3):
                    results.append(run_robustness_one(scenario, 1500, f"dup_{dup_rate:.2f}", method, seed))
    for kind in OUTLIER_KINDS:
        for method in ("quantile5", "fast_mode", "validated"):
            for seed in range(3):
                results.append(run_robustness_one("step_2bp", 1500, f"outlier_{kind}", method, seed))
    return results


def run_robustness_full() -> list:
    """Duplicate-rate grid + outlier contamination across a broader scenario set, 10 seeds, at
    n=5000 -- several minutes, NOT run by pytest."""
    results = []
    for scenario in ("pure_noise", "step_2bp", "step_5bp", "cauchy_x"):
        for dup_rate in DUP_RATES:
            for method in ("quantile5", "fast_mode", "validated"):
                for seed in range(10):
                    results.append(run_robustness_one(scenario, 5000, f"dup_{dup_rate:.2f}", method, seed))
    for scenario in ("pure_noise", "step_2bp", "step_5bp"):
        for kind in OUTLIER_KINDS:
            for method in ("quantile5", "fast_mode", "validated"):
                for seed in range(10):
                    results.append(run_robustness_one(scenario, 5000, f"outlier_{kind}", method, seed))
    return results


def print_robustness_report(results: list) -> None:
    from collections import defaultdict

    by_key: dict = defaultdict(list)
    for r in results:
        by_key[(r.scenario, r.perturbation, r.method)].append(r)
    print(f"{'scenario':12s} {'perturbation':14s} {'method':10s} {'n':>4s} {'bins (mean+/-std, min-max)':>30s} {'RMSE':>16s}")
    for (scenario, perturbation, method), rows in by_key.items():
        bins = np.array([r.bins for r in rows])
        rmses = np.array([r.rmse for r in rows if np.isfinite(r.rmse)])
        rmse_str = f"{rmses.mean():7.3f}+/-{rmses.std():<6.3f}" if rmses.size else "nan"
        print(
            f"{scenario:12s} {perturbation:14s} {method:10s} {len(rows):4d} "
            f"{bins.mean():6.2f}+/-{bins.std():<6.2f} ({bins.min()}-{bins.max()})".ljust(30 + 60)
            + f" {rmse_str:>16s}"
        )
