"""Adversarial stress suite for the significance-gated ("validated") MDLP split (2026-07-19).

Four scenarios the base suite (``bench_mdlp_validated_split_suite.py``) does not cover, all
deferred from the initial validated-split A/B to a follow-up pass:

  1. ``run_near_boundary_noise_stress`` -- pure-noise data engineered to sit at internal decision
     boundaries of the significance gate (the analytic/permutation-null switch at
     ``analytic_null_min_n()`` rows, and varying candidate-cut-point counts via ``n_classes``) --
     does the empirical false-accept rate stay near nominal ``alpha`` at those boundaries, or does
     it show a systematic spike?
  2. ``run_mrmr_confounder_redundancy_stress`` -- a column that is a PURE confounder (derived from
     a genuinely relevant column, zero independent signal) run through the REAL
     ``mrmr._mrmr_class.MRMR.fit`` (not an isolated column-level binning call) with validated-MDLP
     as the binning backend, checking the redundancy gate deprioritizes it.
  3. ``run_extreme_imbalance_boundary_stress`` -- extreme class imbalance (binary 99.9%/0.1%) and
     extreme-outlier continuous targets, swept across n straddling the analytic-null row-count
     floor, checking for crashes / degenerate output / false-accept-rate bias.
  4. ``run_multi_comparisons_defeat`` -- an adversarial attempt to inflate the TREE-WIDE
     false-discovery rate (not just one node) by maximizing the candidate-cut-point count and
     recursion depth on pure noise, across ``bonferroni`` on/off and several ``n_classes``, to see
     whether the per-node Bonferroni correction can be defeated.

All four use >=30-50 seeds per cell and report full distributions (rate + 95% Wilson CI, not a
single point estimate). Run standalone:
``python -m mlframe.feature_selection.filters._benchmarks.bench_mdlp_adversarial_suite``
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._analytic_mi_null import analytic_null_min_n
from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_validated
from mlframe.feature_selection.filters._benchmarks.bench_mdlp_validated_split_suite import (
    MRMR_BINNING_METHODS,
    _MRMR_HARNESS_FIXED_KWARGS,
)


def _wilson_ci(successes: int, n: int, z: float = 1.959964) -> "tuple[float, float]":
    """Wilson score interval for a binomial rate -- reported alongside every empirical rate below
    instead of a bare point estimate (a raw ``k/n`` at n=40-50 has +/-15pp swings on its own)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - margin) / denom, (center + margin) / denom)


# -----------------------------------------------------------------------------
# 1. Near-boundary noise stress: pure noise, swept across the analytic/permutation
#    switch (``analytic_null_min_n()`` rows) and candidate-count-driving n_classes.
# -----------------------------------------------------------------------------


def scen_pure_noise_multiclass(n: int, n_classes: int, seed: int):
    """x, y both independent draws -- x continuous, y a uniform discrete label in
    ``{0, ..., n_classes-1}``. Any accepted split is a false positive by construction."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rng.integers(0, n_classes, n).astype(np.int64)
    return x, y


@dataclass
class NearBoundaryCell:
    n: int
    n_classes: int
    n_accept: int
    n_seeds: int
    rate: float
    ci_lo: float
    ci_hi: float
    analytic_path_expected: bool


def run_near_boundary_noise_stress(n_seeds: int = 40, alpha: float = 0.05) -> "list[NearBoundaryCell]":
    """Root-node-only test (``max_depth=1``) on pure noise, swept across n values straddling the
    analytic-null floor (``analytic_null_min_n()`` -- 25000 by default) and across ``n_classes``
    (which drives both df and the candidate-cut-point Bonferroni divisor). A well-calibrated gate
    should show an accept rate <= alpha at every cell (Bonferroni is a conservative -- not exact --
    correction, so rates well BELOW alpha are expected and fine; rates systematically ABOVE alpha,
    especially clustered right at the analytic/permutation switch, would be the bug this hunts for).
    """
    min_n = analytic_null_min_n()
    n_grid = sorted({max(200, min_n // 20), max(500, min_n // 4), min_n - 2000, min_n - 200, min_n, min_n + 200, min_n + 2000, min_n * 2})
    results = []
    for n in n_grid:
        for n_classes in (2, 4, 8):
            n_accept = 0
            for seed in range(n_seeds):
                x, y = scen_pure_noise_multiclass(n, n_classes, seed=seed * 1_000_003 + n + n_classes)
                edges = mdlp_bin_edges_validated(x, y, max_depth=1, alpha=alpha, seed=seed)
                if edges.size > 2:  # more than just the -inf/+inf sentinels -> at least one split accepted
                    n_accept += 1
            rate = n_accept / n_seeds
            ci_lo, ci_hi = _wilson_ci(n_accept, n_seeds)
            cells = 2 * n_classes
            analytic_expected = n >= min_n and (n / cells) >= 5.0
            results.append(NearBoundaryCell(n, n_classes, n_accept, n_seeds, rate, ci_lo, ci_hi, analytic_expected))
    return results


def print_near_boundary_report(results: "list[NearBoundaryCell]") -> None:
    print(f"{'n':>8s} {'n_classes':>10s} {'path':>10s} {'accept_rate':>12s} {'95% CI':>18s} {'k/n':>8s}")
    for r in results:
        path = "analytic" if r.analytic_path_expected else "perm"
        print(f"{r.n:8d} {r.n_classes:10d} {path:>10s} {r.rate:12.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}] {r.n_accept:4d}/{r.n_seeds:<4d}")


# -----------------------------------------------------------------------------
# 2. Confounder-vs-true-signal through REAL MRMR redundancy screening.
# -----------------------------------------------------------------------------


def scen_pure_confounder(n: int, seed: int, corr_strength: float = 0.9):
    """``relevant`` genuinely drives ``y``; ``confounder`` is DERIVED from ``relevant`` (correlated
    with it) but carries ZERO independent signal beyond what ``relevant`` already gives -- the
    textbook redundant-feature case MRMR's conditional-MI gate exists to catch. A handful of pure
    noise columns are added so the redundancy decision is made in a realistic multi-column context,
    not a trivial two-column frame."""
    rng = np.random.default_rng(seed)
    relevant = rng.standard_normal(n)
    y = relevant * 3.0 + rng.standard_normal(n) * 0.5
    noise_scale = float(relevant.std()) * (1.0 - corr_strength) * 3.0
    confounder = relevant * 2.0 + 5.0 + rng.standard_normal(n) * noise_scale
    X = pd.DataFrame(
        {
            "relevant": relevant,
            "confounder": confounder,
            "noise1": rng.standard_normal(n),
            "noise2": rng.integers(0, 50, n).astype(np.float64),
            "noise3": rng.standard_normal(n) * 100.0,
        }
    )
    return X, y


@dataclass
class ConfounderResult:
    seed: int
    relevant_selected: bool
    confounder_selected: bool


def run_mrmr_confounder_redundancy_stress(n_seeds: int = 40, n: int = 3000) -> "list[ConfounderResult]":
    """Fits the REAL ``mrmr._mrmr_class.MRMR`` (validated-MDLP binning backend, the production
    default) on ``scen_pure_confounder`` across many seeds and records whether the redundancy gate
    correctly drops the confounder while keeping the relevant column."""
    from typing import Any

    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs: "dict[str, Any]" = dict(_MRMR_HARNESS_FIXED_KWARGS)
    kwargs.update(MRMR_BINNING_METHODS["validated"])
    results = []
    for seed in range(n_seeds):
        X, y = scen_pure_confounder(n, seed=seed)
        model = MRMR(random_state=seed, **kwargs)
        model.fit(X, y)
        selected = set(model.get_feature_names_out()) & set(X.columns)
        results.append(ConfounderResult(seed, "relevant" in selected, "confounder" in selected))
    return results


def print_confounder_report(results: "list[ConfounderResult]") -> None:
    n = len(results)
    rel_hits = sum(r.relevant_selected for r in results)
    conf_hits = sum(r.confounder_selected for r in results)
    rel_lo, rel_hi = _wilson_ci(rel_hits, n)
    conf_lo, conf_hi = _wilson_ci(conf_hits, n)
    print(f"relevant selected:   {rel_hits}/{n} = {rel_hits/n:.3f} [{rel_lo:.3f},{rel_hi:.3f}]")
    print(f"confounder selected: {conf_hits}/{n} = {conf_hits/n:.3f} [{conf_lo:.3f},{conf_hi:.3f}]")


# -----------------------------------------------------------------------------
# 3. Extreme-imbalance / extreme-outlier cell-boundary stress.
#
# The analytic-vs-permutation switch (``_split_significant`` in ``_mdlp_validated_split.py``) fires
# when BOTH ``n >= analytic_null_min_n()`` (25000 by default) AND the mean expected cell count
# ``n / (2 * n_classes_full) >= 5.0`` (``_min_expected_cell()``) -- for binary y that second
# condition reduces to ``n >= 20``, so for a 2-class target the row-count floor alone gates the
# switch. Extreme class imbalance stresses this differently: as the recursion isolates the minority
# class into ever-smaller nodes, per-node ``n`` can cross BELOW the floor deep in the tree even when
# the root started well above it -- so a single fit exercises both paths across its own recursion.
# -----------------------------------------------------------------------------


def scen_extreme_imbalance_noise(n: int, minority_frac: float, seed: int):
    """Pure noise (x independent of y): binary y at an extreme (e.g. 0.1%) minority fraction."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = (rng.random(n) < minority_frac).astype(np.int64)
    return x, y


def scen_extreme_outlier_continuous(n: int, outlier_frac: float, seed: int):
    """Pure noise continuous y (independent of x) with a small fraction of extreme-magnitude
    outliers (1e6x scale) -- stresses the quantile pseudo-classing (``max_y_classes``) that feeds
    the significance gate its class count."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    mask = rng.random(n) < outlier_frac
    y = y.copy()
    y[mask] = y[mask] * 1e6
    return x, y


@dataclass
class ImbalanceCell:
    n: int
    minority_frac: float
    n_accept: int
    n_seeds: int
    rate: float
    ci_lo: float
    ci_hi: float
    n_crashes: int
    n_degenerate: int  # n_classes_full <= 1 at the root -> immediate no-op, not a bug, but tracked


def run_extreme_imbalance_boundary_stress(n_seeds: int = 40) -> "list[ImbalanceCell]":
    min_n = analytic_null_min_n()
    n_grid = sorted({min_n - 5000, min_n - 200, min_n, min_n + 200, min_n + 5000})
    results = []
    for n in n_grid:
        for minority_frac in (0.001, 0.01, 0.5):
            n_accept = 0
            n_crashes = 0
            n_degenerate = 0
            for seed in range(n_seeds):
                x, y = scen_extreme_imbalance_noise(n, minority_frac, seed=seed * 7919 + n)
                if int(np.unique(y).size) <= 1:
                    n_degenerate += 1
                    continue
                try:
                    edges = mdlp_bin_edges_validated(x, y, alpha=0.05, seed=seed)
                except Exception as e:  # noqa: BLE001 -- deliberately catching ANY exception, a crash under stress is the bug being hunted
                    n_crashes += 1
                    print(f"CRASH at n={n} minority_frac={minority_frac} seed={seed}: {type(e).__name__}: {e}")
                    continue
                if edges.size > 2:
                    n_accept += 1
            denom = n_seeds - n_degenerate
            rate = n_accept / denom if denom else float("nan")
            ci_lo, ci_hi = _wilson_ci(n_accept, denom) if denom else (float("nan"), float("nan"))
            results.append(ImbalanceCell(n, minority_frac, n_accept, n_seeds, rate, ci_lo, ci_hi, n_crashes, n_degenerate))
    return results


@dataclass
class OutlierCell:
    n: int
    outlier_frac: float
    n_accept: int
    n_seeds: int
    rate: float
    ci_lo: float
    ci_hi: float
    n_crashes: int
    n_nonfinite: int


def run_extreme_outlier_boundary_stress(n_seeds: int = 40) -> "list[OutlierCell]":
    from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges

    min_n = analytic_null_min_n()
    n_grid = sorted({min_n - 200, min_n, min_n + 5000})
    results = []
    for n in n_grid:
        for outlier_frac in (0.001, 0.01, 0.05):
            n_accept = 0
            n_crashes = 0
            n_nonfinite = 0
            for seed in range(n_seeds):
                x, y = scen_extreme_outlier_continuous(n, outlier_frac, seed=seed * 104729 + n)
                try:
                    edges = mdlp_bin_edges(x, y, fast_mode=False)
                except Exception as e:  # noqa: BLE001 -- crash-hunting, any exception is the signal
                    n_crashes += 1
                    print(f"CRASH at n={n} outlier_frac={outlier_frac} seed={seed}: {type(e).__name__}: {e}")
                    continue
                inner = edges[1:-1]
                if inner.size and not np.all(np.isfinite(inner)):
                    n_nonfinite += 1
                if edges.size > 2:
                    n_accept += 1
            rate = n_accept / n_seeds
            ci_lo, ci_hi = _wilson_ci(n_accept, n_seeds)
            results.append(OutlierCell(n, outlier_frac, n_accept, n_seeds, rate, ci_lo, ci_hi, n_crashes, n_nonfinite))
    return results


def print_imbalance_report(results: "list[ImbalanceCell]") -> None:
    print(f"{'n':>8s} {'minority':>9s} {'accept_rate':>12s} {'95% CI':>18s} {'crashes':>8s} {'degenerate':>10s}")
    for r in results:
        print(f"{r.n:8d} {r.minority_frac:9.4f} {r.rate:12.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}] {r.n_crashes:8d} {r.n_degenerate:10d}")


def print_outlier_report(results: "list[OutlierCell]") -> None:
    print(f"{'n':>8s} {'outlier_frac':>13s} {'accept_rate':>12s} {'95% CI':>18s} {'crashes':>8s} {'nonfinite':>10s}")
    for r in results:
        print(f"{r.n:8d} {r.outlier_frac:13.4f} {r.rate:12.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}] {r.n_crashes:8d} {r.n_nonfinite:10d}")


# -----------------------------------------------------------------------------
# 4. Deliberate multi-comparisons-defeat attempt: tree-wide (not single-node) false
#    discovery rate on pure noise, adversarially maximizing recursion depth and
#    candidate-cut-point count.
# -----------------------------------------------------------------------------


@dataclass
class FDRCell:
    n: int
    n_classes: int
    bonferroni: bool
    n_seeds: int
    n_trees_with_fp: int
    tree_fdr: float
    tree_fdr_ci: "tuple[float, float]"
    mean_splits_per_tree: float
    max_splits_in_a_tree: int


def run_multi_comparisons_defeat(n_seeds: int = 50) -> "list[FDRCell]":
    """Full recursion (not root-only) on pure noise, across n / n_classes / bonferroni on-off, to
    hunt for a tree-wide false-discovery rate materially above the single-node nominal alpha=0.05.
    ``n_classes`` is swept up to 8 specifically because a higher class count raises both the
    candidate-cut-point count feeding the mandatory per-node Bonferroni AND the number of children
    each accepted split spawns -- the two levers an adversary would pull to inflate the tree-wide
    rate."""
    results = []
    for n in (1000, 5000, 20000, 50000):
        for n_classes in (2, 4, 8):
            for bonferroni in (False, True):
                n_trees_with_fp = 0
                splits_per_tree = []
                for seed in range(n_seeds):
                    x, y = scen_pure_noise_multiclass(n, n_classes, seed=seed * 15485863 + n + n_classes)
                    edges = mdlp_bin_edges_validated(x, y, alpha=0.05, bonferroni=bonferroni, seed=seed)
                    n_splits = max(0, edges.size - 2)
                    splits_per_tree.append(n_splits)
                    if n_splits > 0:
                        n_trees_with_fp += 1
                tree_fdr = n_trees_with_fp / n_seeds
                ci = _wilson_ci(n_trees_with_fp, n_seeds)
                results.append(
                    FDRCell(
                        n, n_classes, bonferroni, n_seeds, n_trees_with_fp, tree_fdr, ci,
                        float(np.mean(splits_per_tree)), int(np.max(splits_per_tree)),
                    )
                )
    return results


def print_fdr_report(results: "list[FDRCell]") -> None:
    print(f"{'n':>8s} {'n_classes':>10s} {'bonferroni':>11s} {'tree_FDR':>10s} {'95% CI':>18s} {'mean_splits':>12s} {'max_splits':>11s}")
    for r in results:
        print(
            f"{r.n:8d} {r.n_classes:10d} {str(r.bonferroni):>11s} {r.tree_fdr:10.3f} "
            f"[{r.tree_fdr_ci[0]:.3f},{r.tree_fdr_ci[1]:.3f}] {r.mean_splits_per_tree:12.3f} {r.max_splits_in_a_tree:11d}"
        )


def run_adversarial_suite() -> dict:
    """Entry point: runs all 4 scenarios and returns their raw results keyed by name (also used by
    the fast pytest to sanity-check every code path runs without raising)."""
    return {
        "near_boundary": run_near_boundary_noise_stress(),
        "confounder": run_mrmr_confounder_redundancy_stress(),
        "imbalance": run_extreme_imbalance_boundary_stress(),
        "outlier": run_extreme_outlier_boundary_stress(),
        "multi_comparisons": run_multi_comparisons_defeat(),
    }


if __name__ == "__main__":
    results = run_adversarial_suite()
    print("\n=== 1. Near-boundary noise stress (root-node false-accept rate) ===")
    print_near_boundary_report(results["near_boundary"])
    print("\n=== 2. MRMR confounder-vs-relevant redundancy stress ===")
    print_confounder_report(results["confounder"])
    print("\n=== 3a. Extreme class-imbalance boundary stress ===")
    print_imbalance_report(results["imbalance"])
    print("\n=== 3b. Extreme-outlier continuous-target boundary stress ===")
    print_outlier_report(results["outlier"])
    print("\n=== 4. Multi-comparisons-defeat (tree-wide FDR on pure noise) ===")
    print_fdr_report(results["multi_comparisons"])
