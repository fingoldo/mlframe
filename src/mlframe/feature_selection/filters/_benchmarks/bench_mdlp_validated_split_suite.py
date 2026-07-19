"""Recall/precision/F1 comparison of three feature-binning strategies feeding MRMR.fit():

* ``quantile``       -- unsupervised equal-frequency baseline (``nbins_strategy=None``).
* ``mdlp``           -- classic Fayyad-Irani MDLP, njit ("fast") backend.
* ``mdlp_validated`` -- holdout-validated MDLP (``_mdlp_validated_split.mdlp_bin_edges_validated``),
  which only accepts a split that also reduces entropy on a held-out slice; the motivating
  hypothesis is that this suppresses spurious bins MDLP fits to noise on IRRELEVANT columns,
  giving MRMR a cleaner MI signal and better precision without losing recall on real signal.

Ground truth is synthetic (``scen_multicolumn``): each scenario knows exactly which columns are
relevant, redundant (copies/near-copies of a relevant column), or pure irrelevant noise, so
selection quality is scored as recall/precision/F1 against "any relevant column represented in
the selected set" (a redundant column standing in for its relevant twin still counts as a hit --
MRMR is explicitly allowed to prefer the less-noisy copy).

Run modes:
  * ``pytest`` -- ``test_fast_subset`` (a handful of seeds/configs, <10s) is collected normally.
  * ``python bench_mdlp_validated_split_suite.py`` -- full sweep (all configs x 20 seeds), prints
    a recall/precision/F1 table per (strategy, n_relevant) cell and writes nothing to disk.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR

STRATEGIES: "dict[str, dict[str, Any]]" = {
    "quantile": dict(nbins_strategy=None, quantization_method="quantile"),
    "mdlp": dict(nbins_strategy="mdlp", quantization_method="quantile", nbins_strategy_kwargs={"backend": "njit"}),
    "mdlp_validated": dict(nbins_strategy="mdlp_validated", quantization_method="quantile"),
}


@dataclass
class MulticolumnScenario:
    X: pd.DataFrame
    y: pd.Series
    relevant_cols: "set[str]"
    redundant_of: "dict[str, str]"  # redundant col name -> the relevant col it duplicates
    irrelevant_cols: "set[str]" = field(default_factory=set)


def scen_multicolumn(
    n_rows: int = 1200,
    n_relevant: int = 4,
    n_redundant: int = 2,
    n_irrelevant: int = 12,
    seed: int = 0,
    nan_frac: float = 0.05,
) -> MulticolumnScenario:
    """Synthetic classification scenario with known relevant/redundant/irrelevant ground truth.

    Columns are deliberately heterogeneous in distribution/scale/cardinality/NaN so a strategy that
    only works on "nice" Gaussian data doesn't get an easy pass:
      * relevant_i: mixed distributions (normal / lognormal / uniform / low-cardinality integer),
        each individually informative about y at a different effect size (later indices weaker).
      * redundant_i: a noisy copy of a randomly chosen relevant column (correlated, not identical).
      * irrelevant_i: pure noise, same distribution variety as the relevant columns, so a binning
        strategy can't cheat by keying off "looks like signal" shape alone.

    NaN is injected (``nan_frac`` of cells, relevant and irrelevant columns alike) since MDLP's NaN
    handling differs from quantile discretization's -- a validated benchmark has to exercise that path.
    """
    rng = np.random.default_rng(seed)
    n = n_rows

    def _make_col(kind: int) -> np.ndarray:
        if kind == 0:
            return rng.normal(size=n)
        elif kind == 1:
            return rng.lognormal(sigma=1.0, size=n)
        elif kind == 2:
            return rng.uniform(-5, 5, size=n)
        else:
            return rng.integers(0, 6, size=n).astype(np.float64)  # low-cardinality integer

    cols: dict = {}
    relevant_cols: "set[str]" = set()
    redundant_of: dict = {}
    irrelevant_cols: "set[str]" = set()

    logit = np.zeros(n)
    for i in range(n_relevant):
        name = f"relevant_{i}"
        col = _make_col(i % 4)
        cols[name] = col
        relevant_cols.add(name)
        # Effect size decays with i so later relevant columns are weaker/harder to detect --
        # exercises the precision/recall tradeoff instead of an all-or-nothing signal.
        weight = 1.3 / (1.0 + 0.6 * i)
        z = (col - np.nanmean(col)) / (np.nanstd(col) + 1e-9)
        logit += weight * z

    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(np.int64)

    relevant_names = list(relevant_cols)
    for j in range(n_redundant):
        base_name = relevant_names[j % len(relevant_names)]
        base_col = cols[base_name]
        noisy = base_col + rng.normal(scale=0.5 * (np.nanstd(base_col) + 1e-9), size=n)
        name = f"redundant_{j}"
        cols[name] = noisy
        redundant_of[name] = base_name

    for k in range(n_irrelevant):
        name = f"irrelevant_{k}"
        cols[name] = _make_col(k % 4)
        irrelevant_cols.add(name)

    X = pd.DataFrame(cols)
    if nan_frac > 0:
        mask = rng.uniform(size=X.shape) < nan_frac
        X = X.mask(mask)

    return MulticolumnScenario(
        X=X, y=pd.Series(y, name="target"),
        relevant_cols=relevant_cols, redundant_of=redundant_of, irrelevant_cols=irrelevant_cols,
    )


def _score_selection(scen: MulticolumnScenario, selected: "set[str]") -> "tuple[float, float, float]":
    """Recall/precision/F1 against ground truth.

    A selected redundant column counts as a hit on its underlying relevant column (MRMR is
    correctly rewarded, not punished, for picking the less-noisy of two correlated signal
    columns). Precision penalises any selected irrelevant column.
    """
    covered_relevant = set()
    for s in selected:
        if s in scen.relevant_cols:
            covered_relevant.add(s)
        elif s in scen.redundant_of:
            covered_relevant.add(scen.redundant_of[s])
    n_relevant = len(scen.relevant_cols)
    recall = len(covered_relevant) / n_relevant if n_relevant else 0.0
    n_selected = len(selected)
    n_true_positive_selections = sum(1 for s in selected if s in scen.relevant_cols or s in scen.redundant_of)
    precision = n_true_positive_selections / n_selected if n_selected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1


def run_one(scen: MulticolumnScenario, strategy: str) -> "tuple[float, float, float]":
    kwargs = STRATEGIES[strategy]
    m = MRMR(verbose=False, **kwargs)
    m.fit(scen.X, scen.y)
    mask = m.get_support(indices=False)
    selected = {c for c, keep in zip(scen.X.columns, mask) if keep}
    return _score_selection(scen, selected)


def sweep(
    n_relevant_list=(2, 4, 8, 16),
    n_seeds: int = 20,
    n_rows: int = 1200,
    n_redundant: int = 2,
    n_irrelevant: int = 12,
) -> "dict[tuple[str, int], dict[str, float]]":
    """Run every (strategy, n_relevant) cell over ``n_seeds`` seeds; returns mean recall/precision/F1."""
    results: dict = {}
    for n_relevant in n_relevant_list:
        per_strategy_scores: dict = {s: [] for s in STRATEGIES}
        for seed in range(n_seeds):
            scen = scen_multicolumn(
                n_rows=n_rows, n_relevant=n_relevant, n_redundant=n_redundant,
                n_irrelevant=n_irrelevant, seed=seed,
            )
            for strategy in STRATEGIES:
                r, p, f1 = run_one(scen, strategy)
                per_strategy_scores[strategy].append((r, p, f1))
        for strategy, scores in per_strategy_scores.items():
            arr = np.asarray(scores)
            results[(strategy, n_relevant)] = dict(
                recall=float(arr[:, 0].mean()), precision=float(arr[:, 1].mean()), f1=float(arr[:, 2].mean()),
            )
    return results


def print_table(results: "dict[tuple[str, int], dict[str, float]]") -> None:
    n_relevant_vals = sorted({k[1] for k in results})
    for n_relevant in n_relevant_vals:
        print(f"\n-- n_relevant={n_relevant} --")
        print(f"{'strategy':<16}{'recall':>10}{'precision':>12}{'f1':>10}")
        for strategy in STRATEGIES:
            r = results[(strategy, n_relevant)]
            print(f"{strategy:<16}{r['recall']:>10.3f}{r['precision']:>12.3f}{r['f1']:>10.3f}")


def test_fast_subset() -> None:
    """Pytest-collected smoke check: one small config, real assertions.

    MRMR.fit() is not cheap (per-pair CMI / FE screening dominate even at n=300), so this stays to
    a single (n_relevant, seed-count) cell rather than a grid -- the full recall/precision/F1 grid
    lives in ``full_sweep`` / the ``__main__`` block. Asserts recall/precision/F1 are all in [0, 1]
    (sane), and that every strategy beats a trivial zero -- catches a wired-through strategy that
    silently selects nothing (e.g. a broken kwargs plumb producing all-NaN edges).
    """
    results = sweep(n_relevant_list=(2,), n_seeds=2, n_rows=300, n_redundant=1, n_irrelevant=4)
    for (strategy, n_relevant), scores in results.items():
        for metric_name, value in scores.items():
            assert 0.0 <= value <= 1.0, f"{strategy}/{n_relevant}/{metric_name}={value} out of [0,1]"
    for strategy in STRATEGIES:
        r = results[(strategy, 2)]
        assert r["recall"] > 0.0, f"{strategy} selected zero relevant columns on the easiest (n_relevant=2) config"
        assert r["f1"] > 0.0, f"{strategy} F1==0 on the easiest (n_relevant=2) config"


def full_sweep() -> "dict[tuple[str, int], dict[str, float]]":
    """All configs x 20 seeds -- the full benchmark, minutes not seconds. Not pytest-collected."""
    n_seeds = int(os.environ.get("MLFRAME_MDLP_VALIDATED_N_SEEDS", "20"))
    return sweep(n_relevant_list=(2, 4, 8, 16), n_seeds=n_seeds)


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = full_sweep()
    print_table(results)
    print(f"\nfull sweep wall time: {time.perf_counter() - t0:.1f}s")
