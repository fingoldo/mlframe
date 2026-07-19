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

import math
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_oos_validated
from mlframe.feature_selection.filters._adaptive_nbins import _edges_from_quantiles

# -----------------------------------------------------------------------------
# Scenario generators. Each returns (x, y) as float64 1-D arrays, n rows.
# -----------------------------------------------------------------------------


def scen_pure_noise(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n), rng.standard_normal(n) * 1000.0


def scen_step_k_breakpoints(n: int, k: int, seed: int = 0):
    """Step function with exactly ``k`` true breakpoints in [-5, 5], noise sigma=2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n)
    cuts = np.linspace(-5, 5, k + 2)[1:-1]
    levels = rng.uniform(5, 40, k + 1)
    y = np.select([x < c for c in cuts] + [np.ones_like(x, dtype=bool)], [levels[i] for i in range(k)] + [levels[-1]])
    y = y + rng.standard_normal(n) * 2.0
    return x, y


def scen_non_monotonic_sine(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, n)
    y = 20.0 * np.sin(x) + rng.standard_normal(n) * 3.0
    return x, y


def scen_multimodal_target(n: int, seed: int = 0):
    """y bimodal REGARDLESS of x (x carries no signal) -- a distractor multimodal target."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    mode = rng.integers(0, 2, n)
    y = np.where(mode == 0, rng.normal(-50, 5, n), rng.normal(50, 5, n))
    return x, y


def scen_interaction_only(n: int, seed: int = 0):
    """x1 alone carries ZERO marginal signal -- y depends on x1*x2 (XOR-family synergy). A valid
    per-column MDLP criterion should treat x1 like pure noise (only the JOINT with x2 predicts y,
    which is out of scope for a single-column binner)."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.choice([-1.0, 1.0], n)
    y = x1 * x2 * 10.0 + rng.standard_normal(n) * 1.0
    return x1, y  # only x1 passed to the binner -- x2 is the hidden confounder


def scen_lognormal_x(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0, 1.5, n)
    y = np.where(x < 2.0, 10.0, 30.0) + rng.standard_normal(n) * 2.0
    return x, y


def scen_cauchy_x(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_cauchy(n)
    y = np.where(x < 0.0, 10.0, 30.0) + rng.standard_normal(n) * 2.0
    return x, y


def scen_extreme_scale(n: int, seed: int = 0):
    """x spans 1e-3 to 1e6 -- numeric-stability probe."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(1e-3, 1e6, n)
    y = np.where(x < 3e5, 10.0, 30.0) + rng.standard_normal(n) * 2.0
    return x, y


def scen_with_nan(n: int, nan_frac: float, seed: int = 0):
    x, y = scen_step_k_breakpoints(n, k=2, seed=seed)
    rng = np.random.default_rng(seed + 1)
    mask = rng.random(n) < nan_frac
    x = x.copy()
    x[mask] = np.nan
    return x, y


SCENARIOS = {
    "pure_noise": lambda n, seed: scen_pure_noise(n, seed),
    "step_2bp": lambda n, seed: scen_step_k_breakpoints(n, 2, seed),
    "step_5bp": lambda n, seed: scen_step_k_breakpoints(n, 5, seed),
    "step_10bp": lambda n, seed: scen_step_k_breakpoints(n, 10, seed),
    "non_monotonic_sine": lambda n, seed: scen_non_monotonic_sine(n, seed),
    "multimodal_target": lambda n, seed: scen_multimodal_target(n, seed),
    "interaction_only": lambda n, seed: scen_interaction_only(n, seed),
    "lognormal_x": lambda n, seed: scen_lognormal_x(n, seed),
    "cauchy_x": lambda n, seed: scen_cauchy_x(n, seed),
    "extreme_scale_x": lambda n, seed: scen_extreme_scale(n, seed),
    "nan_1pct": lambda n, seed: scen_with_nan(n, 0.01, seed),
    "nan_10pct": lambda n, seed: scen_with_nan(n, 0.10, seed),
    "nan_30pct": lambda n, seed: scen_with_nan(n, 0.30, seed),
}


@dataclass
class Result:
    scenario: str
    n: int
    method: str
    wall: float
    bins: int
    rmse: float
    note: str = ""


def _oos_mse(x_train, y_train, x_test, y_test, edges):
    inner = edges[1:-1] if edges.size >= 2 else edges
    inner = inner[np.isfinite(inner)]
    codes_train = np.searchsorted(inner, x_train, side="right")
    codes_test = np.searchsorted(inner, x_test, side="right")
    n_bins = int(inner.size) + 1
    means = np.full(n_bins, float(np.mean(y_train)) if y_train.size else 0.0)
    for b in range(n_bins):
        m = codes_train == b
        if m.any():
            means[b] = float(np.mean(y_train[m]))
    pred = means[np.clip(codes_test, 0, n_bins - 1)]
    valid = np.isfinite(pred) & np.isfinite(y_test)
    if not valid.any():
        return float("nan"), n_bins
    return float(np.mean((pred[valid] - y_test[valid]) ** 2)), n_bins


def _split(x, y, seed=0, test_frac=0.25):
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    return idx[n_test:], idx[:n_test]


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


if __name__ == "__main__":
    if "--full" in sys.argv:
        print_report(run_full_sweep())
    else:
        print_report(run_fast_subset())
