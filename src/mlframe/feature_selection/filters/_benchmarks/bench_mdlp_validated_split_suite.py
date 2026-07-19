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
import pandas as pd

from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_oos_validated, mdlp_bin_edges_validated
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


# -----------------------------------------------------------------------------
# Ground-truth MRMR recall/precision/F1 harness.
#
# The RMSE bench above measures binning quality in isolation (one column, one
# OOS-MSE fit). It does NOT tell us whether a binning method's per-column edge
# quality actually changes what MRMR.fit() SELECTS on a multi-column problem
# with known relevant / redundant / irrelevant roles -- MRMR's own MI-based
# relevance gate and Fleuret-style conditional-MI redundancy check could easily
# absorb or amplify a binning difference. This section runs the real MRMR
# estimator (``mrmr._mrmr_class.MRMR.fit``), not an isolated column fit.
# -----------------------------------------------------------------------------


@dataclass
class MulticolumnGT:
    """Ground-truth column roles for ``scen_multicolumn``."""

    relevant: list = field(default_factory=list)
    irrelevant: list = field(default_factory=list)
    redundant: list = field(default_factory=list)
    redundant_source: dict = field(default_factory=dict)  # redundant col name -> source relevant col name


def scen_multicolumn(n: int, n_relevant: int, n_irrelevant: int, n_redundant: int, seed: int = 0):
    """Multi-column synthetic with KNOWN ground truth, deliberately non-uniform: relevant columns
    alternate continuous-at-varying-scale / uniform / low-card categorical-coded / high-card
    categorical-coded; irrelevant columns are pure noise in the same style mix; redundant columns
    are derived from a relevant column via one of 3 transforms (additive-noise, affine, signed-log)
    at a randomized correlation strength so the redundant SET has varied correlation structure, not
    one fixed rho. ~30% of columns get NaNs injected (1-15% of rows) to stress the real NaN contract.
    Returns (X: pd.DataFrame, y: np.ndarray, gt: MulticolumnGT)."""
    rng = np.random.default_rng(seed)
    relevant_names = [f"rel_{i}" for i in range(n_relevant)]
    relevant_data: dict = {}
    weights = rng.uniform(0.5, 2.0, n_relevant) * rng.choice([-1.0, 1.0], n_relevant)
    y = np.zeros(n)
    for i, name in enumerate(relevant_names):
        kind = i % 4
        if kind == 0:
            x = rng.standard_normal(n) * (10.0 ** rng.uniform(-1, 3))
        elif kind == 1:
            x = rng.uniform(0.0, 1.0, n) * 100.0
        elif kind == 2:
            x = rng.integers(0, 5, n).astype(np.float64)  # low-cardinality categorical
        else:
            x = rng.integers(0, 500, n).astype(np.float64)  # high-cardinality categorical
        relevant_data[name] = x
        xz = (x - x.mean()) / (x.std() + 1e-9)
        y += weights[i] * xz
    y += rng.standard_normal(n) * 0.5

    cols: dict = dict(relevant_data)

    irrelevant_names = [f"noise_{i}" for i in range(n_irrelevant)]
    for i, name in enumerate(irrelevant_names):
        kind = i % 3
        if kind == 0:
            cols[name] = rng.standard_normal(n) * (10.0 ** rng.uniform(-2, 4))
        elif kind == 1:
            cols[name] = rng.integers(0, 8, n).astype(np.float64)
        else:
            cols[name] = rng.integers(0, 1000, n).astype(np.float64)

    redundant_names = []
    redundant_source: dict = {}
    for i in range(n_redundant):
        src = relevant_names[i % n_relevant]
        base = relevant_data[src]
        base_std = float(base.std()) + 1e-9
        name = f"redund_{i}_of_{src}"
        corr_strength = rng.uniform(0.5, 0.98)  # varied correlation structure across the redundant set
        noise_scale = base_std * (1.0 - corr_strength) * 3.0
        transform = i % 3
        if transform == 0:
            derived = base + rng.standard_normal(n) * noise_scale
        elif transform == 1:
            derived = base * 2.5 + 7.0 + rng.standard_normal(n) * noise_scale
        else:
            derived = np.sign(base) * np.log1p(np.abs(base)) + rng.standard_normal(n) * noise_scale
        cols[name] = derived
        redundant_names.append(name)
        redundant_source[name] = src

    for name in list(cols.keys()):
        if rng.random() < 0.3:
            frac = rng.uniform(0.01, 0.15)
            mask = rng.random(n) < frac
            arr = cols[name].astype(np.float64).copy()
            arr[mask] = np.nan
            cols[name] = arr

    X = pd.DataFrame(cols)
    gt = MulticolumnGT(relevant=relevant_names, irrelevant=irrelevant_names, redundant=redundant_names, redundant_source=redundant_source)
    return X, y, gt


# nbins_strategy / nbins_strategy_kwargs to feed MRMR() per binning method under test (mirrors the
# ``method`` values used by ``run_one`` above, minus ``oos_validated`` -- the OOS-validated variant
# is not wired into ``supervised_binning.mdlp_bin_edges`` / MRMR's ``nbins_strategy`` dispatch, it is
# a standalone alternative function, so it has no MRMR-selectable analogue here).
MRMR_BINNING_METHODS: "dict[str, dict]" = {
    "quantile5": {"nbins_strategy": None, "quantization_nbins": 10},
    "fast_mode": {"nbins_strategy": "mdlp", "nbins_strategy_kwargs": {"mdlp_fast_mode": True}},
    "validated": {"nbins_strategy": "mdlp", "nbins_strategy_kwargs": {"mdlp_fast_mode": False}},
}

# Kept off/minimal so the harness measures the BINNING method's effect on selection, not FE/cluster-
# aggregate machinery layered on top; ``use_simple_mode=False`` (the MRMR default) is kept ON because
# disabling it skips the conditional-MI redundancy check entirely, which would make the redundant-
# column false-positive-rate metric meaningless (everything redundant would pass through unchecked).
_MRMR_HARNESS_FIXED_KWARGS = dict(
    cluster_aggregate_enable=False,
    dcd_enable=False,
    fe_unary_preset="minimal",
    fe_binary_preset="minimal",
    fe_max_steps=1,
    verbose=0,
    n_jobs=1,
)


def run_mrmr_selection(X: pd.DataFrame, y: np.ndarray, method: str, random_seed: int = 0):
    """Fit the real ``mrmr._mrmr_class.MRMR`` (not an isolated column-level binning call) with the
    given binning method wired in, and return the set of ORIGINAL (non-engineered) column names it
    selected. Engineered/composite feature names from FE are excluded from the returned set (FE is
    held at 'minimal' + 1 step precisely so it stays a minor factor, not the object under test) --
    they are neither counted as a ground-truth hit nor a false positive."""
    from typing import Any

    from mlframe.feature_selection.filters.mrmr import MRMR

    if method not in MRMR_BINNING_METHODS:
        raise ValueError(method)
    kwargs: "dict[str, Any]" = dict(_MRMR_HARNESS_FIXED_KWARGS)
    kwargs.update(MRMR_BINNING_METHODS[method])
    kwargs["random_seed"] = random_seed
    model = MRMR(**kwargs)
    model.fit(X, y)
    selected = set(model.get_feature_names_out())
    return selected & set(X.columns)


@dataclass
class MrmrGTResult:
    config: str
    method: str
    seed: int
    recall: float
    precision: float
    fpr_noise_redundant: float
    f1: float
    n_selected: int


def _prf(selected: set, gt: MulticolumnGT) -> tuple:
    relevant = set(gt.relevant)
    noise_and_redundant = set(gt.irrelevant) | set(gt.redundant)
    hits = selected & relevant
    recall = len(hits) / len(relevant) if relevant else float("nan")
    precision = len(hits) / len(selected) if selected else 0.0
    fpr = len(selected & noise_and_redundant) / len(noise_and_redundant) if noise_and_redundant else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, fpr, f1


def run_mrmr_gt_config(n: int, n_relevant: int, n_irrelevant: int, n_redundant: int, methods, seeds, config_label: str) -> list:
    results = []
    for seed in seeds:
        X, y, gt = scen_multicolumn(n, n_relevant, n_irrelevant, n_redundant, seed=seed)
        for method in methods:
            selected = run_mrmr_selection(X, y, method, random_seed=seed)
            recall, precision, fpr, f1 = _prf(selected, gt)
            results.append(MrmrGTResult(config_label, method, seed, recall, precision, fpr, f1, len(selected)))
    return results


def run_mrmr_fast_subset() -> list:
    """Small-n, few-seed MRMR ground-truth sweep -- a few seconds total. Used by the pytest fast
    test. Covers 2/4/8 relevant columns (16 is exercised only in ``run_mrmr_full_sweep``, it needs a
    bigger n to keep any signal detectable through the noise floor at this row count)."""
    results = []
    for n_relevant in (2, 4, 8):
        results.extend(
            run_mrmr_gt_config(
                n=800,
                n_relevant=n_relevant,
                n_irrelevant=4,
                n_redundant=n_relevant,
                methods=("quantile5", "fast_mode", "validated"),
                seeds=range(5),
                config_label=f"rel{n_relevant}",
            )
        )
    return results


def run_mrmr_full_sweep() -> list:
    """Thorough MRMR ground-truth sweep: 2/4/8/16 relevant columns, larger n, 20 seeds/config,
    all 3 binning methods -- several minutes, NOT run by pytest."""
    results = []
    for n_relevant in (2, 4, 8, 16):
        n = 4000 if n_relevant <= 8 else 12000
        results.extend(
            run_mrmr_gt_config(
                n=n,
                n_relevant=n_relevant,
                n_irrelevant=max(8, n_relevant),
                n_redundant=n_relevant,
                methods=("quantile5", "fast_mode", "validated"),
                seeds=range(20),
                config_label=f"rel{n_relevant}",
            )
        )
    return results


def print_mrmr_gt_report(results: list) -> None:
    from collections import defaultdict

    by_key: dict = defaultdict(list)
    for r in results:
        by_key[(r.config, r.method)].append(r)
    print(f"{'config':10s} {'method':12s} {'n':>4s} {'recall':>16s} {'precision':>16s} {'fpr':>16s} {'f1':>16s}")
    for (config, method), rows in by_key.items():
        recalls = np.array([r.recall for r in rows])
        precisions = np.array([r.precision for r in rows])
        fprs = np.array([r.fpr_noise_redundant for r in rows])
        f1s = np.array([r.f1 for r in rows])
        print(
            f"{config:10s} {method:12s} {len(rows):4d} "
            f"{recalls.mean():7.3f}+/-{recalls.std():<6.3f} "
            f"{precisions.mean():7.3f}+/-{precisions.std():<6.3f} "
            f"{fprs.mean():7.3f}+/-{fprs.std():<6.3f} "
            f"{f1s.mean():7.3f}+/-{f1s.std():<6.3f}"
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
    """2 scenarios (signal + pure-noise) x reduced grid x 3 seeds at n=1500 -- a few seconds."""
    results = []
    for scenario in ("step_2bp", "pure_noise"):
        for alpha in (0.01, 0.2):  # endpoints only -- the full 4-point grid is a full-sweep concern
            for n_permutations in (50, 1000):
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


# -----------------------------------------------------------------------------
# Duplicate-row and outlier/heavy-tail robustness.
#
# Exact duplicate rows inflate the raw row count the significance test sees without adding
# independent evidence; check whether this causes over-splitting relative to the same rows
# without duplication (matched information content). Outlier contamination is checked separately
# for a degenerate/wrong-result failure mode. Compares validated vs classic fast_mode vs the
# quantile baseline, multi-seed, distributions reported (not just means).
# -----------------------------------------------------------------------------


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


if __name__ == "__main__":
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
