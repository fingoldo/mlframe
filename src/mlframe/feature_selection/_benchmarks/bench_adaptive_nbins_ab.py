"""A/B benchmark for adaptive nbins WAVE 1 fixes (2026-05-29).

For every (distribution x signal x N x fold) cell, evaluates EACH method
under multiple TREATMENT variants:

  * ``baseline``           - all fixes OFF, matches the 2026-05-29 leaderboard.
  * ``mm_on``              - Miller-Madow bias correction in ``_plug_in_mi``.
  * ``knuth_quantile``     - Knuth optimum M + QUANTILE edges (vs uniform).
  * ``knuth_mmax64``       - Knuth M_max capped at 64 (vs sqrt(N)*4 up to 500).
  * ``knuth_combo``        - Knuth: quantile + M_max=64.
  * ``bb_midpoint``        - BB backtrack uses cell-boundary midpoints (Scargle/astropy convention).
  * ``bb_p010``            - BB p0=0.10 (vs 0.05 astropy time-series default).
  * ``bb_subsample1000``   - BB sub-samples to <=1000 points before DP.
  * ``bb_combo``           - BB: midpoint + p0=0.10 + subsample=1000.
  * ``mdlp_njit``          - MDLP backend='njit' (10-30x speedup target).
  * ``mdlp_scaled``        - MDLP scaled_min_split: max(5, 0.02*N).
  * ``mdlp_combo``         - MDLP: njit + scaled.

The ``mm_on`` treatment is plug-in scoring-side, so it propagates to ALL
methods uniformly; the other treatments are per-method.

Output: per-(method, treatment) leaderboard with deltas vs baseline.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mlframe.feature_selection.filters._adaptive_nbins import (
    per_feature_edges,
    _plug_in_mi,
)
from mlframe.feature_selection._benchmarks.bench_adaptive_nbins import (
    _draw_distribution,
    _draw_signal,
    _make_folds,
    _legacy_quantile_edges,
    _bin_with_edges,
)


# =============================================================================
# Treatment definitions
# =============================================================================


# Each treatment specifies (a) method_kwargs passed to per_feature_edges
# scoped by method name, and (b) whether Miller-Madow is on during scoring.
TREATMENTS: Dict[str, Dict] = {
    "baseline": {
        "mm": False,
        "kwargs_by_method": {},
    },
    "mm_on": {
        "mm": True,
        "kwargs_by_method": {},
    },
    "knuth_quantile": {
        "mm": False,
        "kwargs_by_method": {"knuth": {"knuth_edge_type": "quantile"}},
    },
    "knuth_mmax64": {
        "mm": False,
        "kwargs_by_method": {"knuth": {"knuth_m_max_cap": 64}},
    },
    "knuth_combo": {
        "mm": False,
        "kwargs_by_method": {
            "knuth": {"knuth_edge_type": "quantile", "knuth_m_max_cap": 64}
        },
    },
    "bb_midpoint": {
        "mm": False,
        "kwargs_by_method": {"blocks": {"bb_edge_placement": "midpoint"}},
    },
    "bb_p010": {
        "mm": False,
        "kwargs_by_method": {"blocks": {"p0": 0.10}},
    },
    "bb_subsample1000": {
        "mm": False,
        "kwargs_by_method": {"blocks": {"bb_subsample_threshold": 1000}},
    },
    "bb_combo": {
        "mm": False,
        "kwargs_by_method": {
            "blocks": {
                "bb_edge_placement": "midpoint",
                "p0": 0.10,
                "bb_subsample_threshold": 1000,
            }
        },
    },
    "mdlp_njit": {
        "mm": False,
        "kwargs_by_method": {"fayyad_irani": {"mdlp_backend": "njit"}},
    },
    "mdlp_scaled": {
        "mm": False,
        "kwargs_by_method": {"fayyad_irani": {"mdlp_scaled_min_split": True}},
    },
    "mdlp_combo": {
        "mm": False,
        "kwargs_by_method": {
            "fayyad_irani": {"mdlp_backend": "njit", "mdlp_scaled_min_split": True}
        },
    },
    "ALL_FIXES": {  # the bundled patch the user would ship as a single new default
        "mm": True,
        "kwargs_by_method": {
            "knuth": {"knuth_edge_type": "quantile", "knuth_m_max_cap": 64},
            "blocks": {
                "bb_edge_placement": "midpoint",
                "p0": 0.10,
                "bb_subsample_threshold": 1000,
            },
            "fayyad_irani": {"mdlp_backend": "njit", "mdlp_scaled_min_split": True},
        },
    },
}


# =============================================================================
# Fold scorer
# =============================================================================


@dataclass
class ABFoldResult:
    method: str
    treatment: str
    distribution: str
    signal_kind: str
    n: int
    fold_idx: int
    mi_val: float
    nbins: int
    runtime_ms: float


METHODS_BASE = [
    "quantile10",
    "sturges",
    "freedman_diaconis",
    "knuth",
    "blocks",
    "fayyad_irani",
    "optimal_joint",
]


def _treatment_applies(method: str, treatment: str) -> bool:
    """Skip irrelevant (method, treatment) combos to keep bench tractable."""
    if treatment in ("baseline", "mm_on", "ALL_FIXES"):
        return True
    spec = TREATMENTS[treatment]
    method_key = {"fayyad_irani": "fayyad_irani", "blocks": "blocks"}.get(method, method)
    return method_key in spec["kwargs_by_method"]


def _run_fold_ab(
    method: str,
    treatment: str,
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    distribution: str, signal_kind: str, n: int, fold_idx: int,
) -> Optional[ABFoldResult]:
    spec = TREATMENTS[treatment]
    method_kwargs = spec["kwargs_by_method"].get(
        # blocks/mdlp method names in dispatcher
        {"fayyad_irani": "fayyad_irani", "blocks": "blocks"}.get(method, method), {}
    )
    mm = spec["mm"]
    X_train = x_train.reshape(-1, 1)
    t0 = time.perf_counter()
    if method == "quantile10":
        edges = _legacy_quantile_edges(x_train, n_bins=10)
    else:
        try:
            edges_list = per_feature_edges(X_train, y_train, method=method, **method_kwargs)
            edges = edges_list[0]
        except Exception:
            return None
    t_ms = (time.perf_counter() - t0) * 1000.0
    val_binned = _bin_with_edges(x_val, edges)
    mi = _plug_in_mi(val_binned, y_val, miller_madow=mm)
    return ABFoldResult(
        method=method, treatment=treatment, distribution=distribution,
        signal_kind=signal_kind, n=n, fold_idx=fold_idx,
        mi_val=float(mi), nbins=int(edges.size + 1), runtime_ms=float(t_ms),
    )


# =============================================================================
# Bench driver
# =============================================================================


def run_ab_benchmark(
    distributions: Optional[List[str]] = None,
    signal_kinds: Optional[List[str]] = None,
    sample_sizes: Optional[List[int]] = None,
    n_splits: int = 5,
    n_repeats: int = 2,
    treatments: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    random_state: int = 0,
    verbose: int = 1,
) -> Dict:
    distributions = distributions or [
        "gaussian", "lognormal", "bimodal", "heavy_tail_t",
        "uniform", "exponential", "discrete_low_card", "near_constant",
    ]
    signal_kinds = signal_kinds or [
        "no_signal", "linear", "monotone", "threshold", "xor", "sin",
    ]
    sample_sizes = sample_sizes or [500, 2000, 10000]
    treatments = treatments or list(TREATMENTS.keys())
    methods = methods or METHODS_BASE

    rng_master = np.random.default_rng(random_state)
    all_results: List[ABFoldResult] = []
    total_folds = (len(distributions) * len(signal_kinds) * len(sample_sizes)
                   * n_repeats * n_splits * len(methods) * len(treatments))
    if verbose:
        print(f"[bench_adaptive_nbins_ab] {total_folds} fold-evaluations "
              f"({len(distributions)}d x {len(signal_kinds)}s x {len(sample_sizes)}n "
              f"x {n_repeats}r x {n_splits}f x {len(methods)}m x {len(treatments)}t)")

    counter = 0
    for dist in distributions:
        for sig in signal_kinds:
            for n in sample_sizes:
                for rep in range(n_repeats):
                    rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
                    x = _draw_distribution(dist, n, rng)
                    y = _draw_signal(sig, x, rng)
                    folds = _make_folds(n, n_splits, rng)
                    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                        x_tr, y_tr = x[tr_idx], y[tr_idx]
                        x_va, y_va = x[va_idx], y[va_idx]
                        for method in methods:
                            for treatment in treatments:
                                if not _treatment_applies(method, treatment):
                                    continue
                                res = _run_fold_ab(
                                    method=method, treatment=treatment,
                                    x_train=x_tr, y_train=y_tr,
                                    x_val=x_va, y_val=y_va,
                                    distribution=dist, signal_kind=sig,
                                    n=n, fold_idx=fold_idx,
                                )
                                if res is not None:
                                    all_results.append(res)
                                counter += 1
                if verbose:
                    print(f"  done: dist={dist:15s} signal={sig:12s} "
                          f"({counter}/{total_folds})")

    summary = _summarise_ab(all_results)
    return {"results": [asdict(r) for r in all_results], "summary": summary}


# =============================================================================
# Aggregation
# =============================================================================


def _summarise_ab(results: List[ABFoldResult]) -> Dict:
    by_mt: Dict[Tuple[str, str], List[ABFoldResult]] = {}
    for r in results:
        by_mt.setdefault((r.method, r.treatment), []).append(r)

    per_mt: Dict[str, Dict[str, Dict]] = {}
    for (m, t), rs in by_mt.items():
        mis = np.array([r.mi_val for r in rs])
        rts = np.array([r.runtime_ms for r in rs])
        nbins = np.array([r.nbins for r in rs])
        per_mt.setdefault(m, {})[t] = {
            "mi_mean": float(mis.mean()),
            "mi_median": float(np.median(mis)),
            "rt_ms_mean": float(rts.mean()),
            "rt_ms_median": float(np.median(rts)),
            "nbins_mean": float(nbins.mean()),
            "n_folds": len(rs),
        }

    # MI on no_signal per (method, treatment).
    no_sig: Dict[str, Dict[str, float]] = {}
    for (m, t), rs in by_mt.items():
        sub = [r.mi_val for r in rs if r.signal_kind == "no_signal"]
        if sub:
            no_sig.setdefault(m, {})[t] = float(np.mean(sub))

    return {"per_method_treatment": per_mt, "mi_no_signal": no_sig}


def print_ab_summary(summary: Dict) -> None:
    pmt = summary["per_method_treatment"]
    no_sig = summary["mi_no_signal"]

    print("\n" + "=" * 88)
    print("BENCH A/B: adaptive_nbins WAVE 1 fixes vs baseline")
    print("=" * 88)

    for method in METHODS_BASE:
        if method not in pmt:
            continue
        print(f"\n[method = {method}]")
        print("-" * 88)
        print(f"{'treatment':<20} {'MI_mean':>9} {'dMI':>8} {'MI_nosig':>9} {'dnosig':>8} "
              f"{'rt_ms':>9} {'drt%':>8}")
        baseline = pmt[method].get("baseline")
        for tname in TREATMENTS.keys():
            if tname not in pmt[method]:
                continue
            t = pmt[method][tname]
            n_s = no_sig.get(method, {}).get(tname, float("nan"))
            b_ns = no_sig.get(method, {}).get("baseline", float("nan"))
            if baseline:
                dmi = t["mi_mean"] - baseline["mi_mean"]
                dns = n_s - b_ns
                drt = (
                    100.0 * (t["rt_ms_mean"] - baseline["rt_ms_mean"])
                    / max(baseline["rt_ms_mean"], 1e-9)
                )
            else:
                dmi = float("nan")
                dns = float("nan")
                drt = float("nan")
            print(f"{tname:<20} {t['mi_mean']:>9.4f} {dmi:>+8.4f} "
                  f"{n_s:>9.4f} {dns:>+8.4f} "
                  f"{t['rt_ms_mean']:>9.3f} {drt:>+8.1f}")
    print()


# =============================================================================
# Entry
# =============================================================================


def main():
    out = run_ab_benchmark(
        sample_sizes=[500, 2000, 10000],
        n_splits=5, n_repeats=2, verbose=1,
    )
    print_ab_summary(out["summary"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path("D:/Temp") / f"bench_adaptive_nbins_ab_{ts}.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[bench_adaptive_nbins_ab] -> {out_path}")
    except Exception as exc:
        print(f"[bench_adaptive_nbins_ab] save failed: {exc!r}")


if __name__ == "__main__":
    main()
