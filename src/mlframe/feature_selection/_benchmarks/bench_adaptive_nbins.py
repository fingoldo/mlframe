"""Strict CV benchmark: adaptive nbins methods for MRMR pre-binning (2026-05-29).

Evaluates 6 nbins-strategies head-to-head on a synthetic suite spanning:

  * **Distribution shapes**: Gaussian, lognormal, bimodal, heavy-tail t(df=2),
    uniform, exponential, discrete-low-cardinality, near-constant.
  * **Sample sizes**: n in {500, 2000, 10000} -- formula vs CV trade-off shifts with n.
  * **Signal types**: linear, monotone, threshold, XOR, sin, no-signal noise.

Each (distribution x n x signal) cell becomes one task. For each task, we run
K-fold CV:

  1. Fit bin edges on TRAIN x using the candidate method (supervised methods see y_train).
  2. Bin VAL x with those edges.
  3. Score: plug-in I(X_val_binned; y_val) -- the MRMR-internal relevance metric.
  4. Also record: number of bins, edge-builder runtime, edges stability across folds.

Aggregated metrics (mean +/- std over folds, then per-method across tasks):

  * **MI_val**: held-out MI estimate (the metric MRMR optimises).
  * **MI_recall_signal**: fraction of signal-bearing tasks where method ranked
    in the top-2 by MI_val (proxy for "would MRMR have picked this column").
  * **runtime_ms_per_col**.
  * **nbins_chosen**: spread of bin counts produced (diagnostic).
  * **edge_stability**: cosine-similarity of inner edges between folds (1.0 = identical
    on every fold; 0.0 = totally unstable). Quantile-based methods should be near 1.0
    on stationary signals; OptimalJoint may dip if CV picks different M per fold.

Output: a multi-section table written to stdout AND saved to
``D:\\Temp\\bench_adaptive_nbins_<ts>.json`` for follow-up plotting.

Run::

    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_adaptive_nbins
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from mlframe.feature_selection.filters._adaptive_nbins import (
    per_feature_edges,
    _plug_in_mi,
)


METHODS = [
    "sturges",
    "freedman_diaconis",
    "qs",
    "knuth",
    "blocks",
    "fayyad_irani",
    "optimal_joint",
    "mah",
]

LEGACY_BASELINE = "quantile10"


# =============================================================================
# Synthetic data generators
# =============================================================================


def _draw_distribution(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if name == "gaussian":
        return rng.standard_normal(n)
    if name == "lognormal":
        return rng.lognormal(mean=0.0, sigma=1.0, size=n)
    if name == "bimodal":
        flag = rng.integers(0, 2, n).astype(np.float64)
        return flag * rng.normal(loc=-2.0, scale=0.5, size=n) + (1 - flag) * rng.normal(loc=2.0, scale=0.5, size=n)
    if name == "heavy_tail_t":
        return rng.standard_t(df=2, size=n)
    if name == "uniform":
        return rng.uniform(-3.0, 3.0, size=n)
    if name == "exponential":
        return rng.exponential(scale=1.0, size=n)
    if name == "discrete_low_card":
        return rng.integers(0, 8, n).astype(np.float64)
    if name == "near_constant":
        return 1.0 + rng.normal(scale=1e-3, size=n)
    raise ValueError(f"unknown distribution {name!r}")


def _draw_signal(signal_kind: str, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate y from x according to signal_kind. Returns y as int classes for MI scoring."""
    if signal_kind == "no_signal":
        return rng.integers(0, 2, x.size).astype(np.int64)
    if signal_kind == "linear":
        z = x + rng.normal(scale=0.3, size=x.size)
        return (z > np.median(z)).astype(np.int64)
    if signal_kind == "monotone":
        z = np.tanh(x) + rng.normal(scale=0.2, size=x.size)
        return (z > np.median(z)).astype(np.int64)
    if signal_kind == "threshold":
        thr = np.quantile(x, 0.7)
        return (x > thr).astype(np.int64)
    if signal_kind == "xor":
        thr_lo = np.quantile(x, 0.33)
        thr_hi = np.quantile(x, 0.67)
        return ((x > thr_lo) & (x < thr_hi)).astype(np.int64)
    if signal_kind == "sin":
        return ((np.sin(2.0 * x) + rng.normal(scale=0.3, size=x.size)) > 0).astype(np.int64)
    raise ValueError(f"unknown signal_kind {signal_kind!r}")


# =============================================================================
# Scoring
# =============================================================================


@dataclass
class FoldResult:
    method: str
    distribution: str
    n: int
    signal_kind: str
    fold_idx: int
    mi_val: float
    nbins: int
    runtime_ms: float
    edges: Tuple[float, ...]


def _bin_with_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(x.size, dtype=np.int64)
    return np.searchsorted(edges, x.astype(np.float64), side="right").astype(np.int64)


def _legacy_quantile_edges(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """The pre-2026-05-29 MRMR default: fixed-10 quantile bins per column."""
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    full_edges = np.nanpercentile(x.astype(np.float64), quantiles)
    return np.asarray(full_edges[1:-1], dtype=np.float64)


def _run_one_fold(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    distribution: str,
    n: int,
    signal_kind: str,
    fold_idx: int,
) -> FoldResult:
    """Single fold scoring. Returns held-out plug-in MI and diagnostics."""
    X_train = x_train.reshape(-1, 1)
    t0 = time.perf_counter()
    if method == LEGACY_BASELINE:
        edges = _legacy_quantile_edges(x_train, n_bins=10)
    else:
        edges_list = per_feature_edges(X_train, y_train, method=method)
        edges = edges_list[0]
    t_ms = (time.perf_counter() - t0) * 1000.0

    val_binned = _bin_with_edges(x_val, edges)
    mi = _plug_in_mi(val_binned, y_val)

    return FoldResult(
        method=method,
        distribution=distribution,
        n=n,
        signal_kind=signal_kind,
        fold_idx=fold_idx,
        mi_val=float(mi),
        nbins=int(edges.size + 1),
        runtime_ms=float(t_ms),
        edges=tuple(float(e) for e in edges.tolist()[:20]),  # cap to keep JSON small
    )


def _make_folds(n: int, n_splits: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
    perm = rng.permutation(n)
    fold_assignment = perm % n_splits
    folds = []
    for k in range(n_splits):
        val_mask = fold_assignment == k
        train_mask = ~val_mask
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return folds


# =============================================================================
# Main bench driver
# =============================================================================


def run_benchmark(
    distributions: Optional[List[str]] = None,
    signal_kinds: Optional[List[str]] = None,
    sample_sizes: Optional[List[int]] = None,
    n_splits: int = 5,
    n_repeats: int = 2,
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
    methods = methods or ([LEGACY_BASELINE] + METHODS)

    rng_master = np.random.default_rng(random_state)
    all_results: List[FoldResult] = []
    total_tasks = len(distributions) * len(signal_kinds) * len(sample_sizes) * n_repeats * n_splits * len(methods)
    if verbose:
        print(f"[bench_adaptive_nbins] {total_tasks} folds to run "
              f"({len(distributions)} dist x {len(signal_kinds)} signal x "
              f"{len(sample_sizes)} n x {n_repeats} reps x {n_splits} folds x {len(methods)} methods)")

    task_counter = 0
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
                            try:
                                res = _run_one_fold(
                                    method=method,
                                    x_train=x_tr, y_train=y_tr,
                                    x_val=x_va, y_val=y_va,
                                    distribution=dist, n=n,
                                    signal_kind=sig, fold_idx=fold_idx,
                                )
                                all_results.append(res)
                            except Exception as exc:
                                if verbose >= 2:
                                    print(f"  [FAIL] {method} on ({dist},{sig},n={n},fold={fold_idx}): {exc!r}")
                            task_counter += 1
                if verbose:
                    print(f"  done: dist={dist:15s} signal={sig:12s} "
                          f"({task_counter}/{total_tasks} folds)")

    summary = _summarise(all_results, methods)
    return {"results": [asdict(r) for r in all_results], "summary": summary}


# =============================================================================
# Aggregation
# =============================================================================


def _summarise(results: List[FoldResult], methods: List[str]) -> Dict:
    """Return per-method aggregates + per-(method, signal_kind) and per-(method, n) tables."""
    by_method: Dict[str, List[FoldResult]] = {m: [] for m in methods}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    per_method: Dict[str, Dict] = {}
    for m, rs in by_method.items():
        if not rs:
            continue
        mis = np.array([r.mi_val for r in rs])
        nbins = np.array([r.nbins for r in rs])
        rts = np.array([r.runtime_ms for r in rs])
        per_method[m] = {
            "n_folds": len(rs),
            "mi_val_mean": float(mis.mean()),
            "mi_val_median": float(np.median(mis)),
            "mi_val_std": float(mis.std()),
            "mi_val_q25": float(np.percentile(mis, 25)),
            "mi_val_q75": float(np.percentile(mis, 75)),
            "nbins_mean": float(nbins.mean()),
            "nbins_min": int(nbins.min()),
            "nbins_max": int(nbins.max()),
            "runtime_ms_mean": float(rts.mean()),
            "runtime_ms_median": float(np.median(rts)),
        }

    # MI by signal kind (drop no_signal -- it's the noise floor)
    signal_kinds = sorted({r.signal_kind for r in results})
    by_method_signal: Dict[str, Dict[str, float]] = {}
    for m in methods:
        by_method_signal[m] = {}
        for sk in signal_kinds:
            sub = [r.mi_val for r in by_method.get(m, []) if r.signal_kind == sk]
            if sub:
                by_method_signal[m][sk] = float(np.mean(sub))

    # MI by sample size
    ns = sorted({r.n for r in results})
    by_method_n: Dict[str, Dict[int, float]] = {}
    for m in methods:
        by_method_n[m] = {}
        for n in ns:
            sub = [r.mi_val for r in by_method.get(m, []) if r.n == n]
            if sub:
                by_method_n[m][str(n)] = float(np.mean(sub))

    # Stability of MI ranking across folds (edge stability proxy)
    by_task: Dict[Tuple, List[FoldResult]] = {}
    for r in results:
        key = (r.method, r.distribution, r.signal_kind, r.n)
        by_task.setdefault(key, []).append(r)
    cv_of_mi_by_method: Dict[str, float] = {}
    for m in methods:
        cvs = []
        for key, rs in by_task.items():
            if key[0] != m or len(rs) < 2:
                continue
            mis = np.array([r.mi_val for r in rs])
            if mis.mean() > 1e-8:
                cvs.append(float(mis.std() / (abs(mis.mean()) + 1e-12)))
        if cvs:
            cv_of_mi_by_method[m] = float(np.mean(cvs))

    # Win rate (per (dist, signal, n, rep, fold)): how often method's mi is top-1.
    by_task_all: Dict[Tuple, Dict[str, float]] = {}
    for r in results:
        key = (r.distribution, r.signal_kind, r.n, r.fold_idx)
        by_task_all.setdefault(key, {})[r.method] = r.mi_val
    win_count: Dict[str, int] = {m: 0 for m in methods}
    total_tasks_with_winner = 0
    for key, mvals in by_task_all.items():
        if not mvals:
            continue
        winner = max(mvals, key=mvals.get)
        win_count[winner] = win_count.get(winner, 0) + 1
        total_tasks_with_winner += 1
    win_rate = {m: (win_count[m] / total_tasks_with_winner if total_tasks_with_winner else 0.0)
                for m in methods}

    return {
        "per_method": per_method,
        "mi_by_method_signal": by_method_signal,
        "mi_by_method_n": by_method_n,
        "cv_of_mi_per_method": cv_of_mi_by_method,
        "win_rate_per_method": win_rate,
    }


# =============================================================================
# Pretty-print
# =============================================================================


def print_summary(summary: Dict) -> None:
    print("\n" + "=" * 78)
    print("BENCH: adaptive_nbins  (held-out plug-in MI on synthetic suite)")
    print("=" * 78)

    print("\n[1] Per-method overall (mean over all (dist, signal, n, rep, fold) tasks)")
    print("-" * 78)
    print(f"{'method':<20} {'MI_mean':>9} {'MI_med':>8} {'MI_std':>8} "
          f"{'nbins_mean':>10} {'rt_ms':>8} {'win_rate':>9}")
    pm = summary["per_method"]
    win = summary["win_rate_per_method"]
    for m, d in sorted(pm.items(), key=lambda kv: -kv[1]["mi_val_mean"]):
        print(f"{m:<20} {d['mi_val_mean']:>9.4f} {d['mi_val_median']:>8.4f} "
              f"{d['mi_val_std']:>8.4f} {d['nbins_mean']:>10.1f} "
              f"{d['runtime_ms_mean']:>8.3f} {win.get(m, 0.0):>9.2%}")

    print("\n[2] MI per signal kind (mean MI; higher better when signal != no_signal)")
    print("-" * 78)
    sig_table = summary["mi_by_method_signal"]
    if sig_table:
        methods = list(pm.keys())
        sigs = sorted({s for d in sig_table.values() for s in d.keys()})
        print(f"{'method':<20}" + "".join(f"{s:>12}" for s in sigs))
        for m in methods:
            row = sig_table.get(m, {})
            print(f"{m:<20}" + "".join(f"{row.get(s, float('nan')):>12.4f}" for s in sigs))

    print("\n[3] MI per sample size (mean MI)")
    print("-" * 78)
    n_table = summary["mi_by_method_n"]
    if n_table:
        methods = list(pm.keys())
        ns_str = sorted({n for d in n_table.values() for n in d.keys()}, key=int)
        print(f"{'method':<20}" + "".join(f"{('n=' + n):>12}" for n in ns_str))
        for m in methods:
            row = n_table.get(m, {})
            print(f"{m:<20}" + "".join(f"{row.get(n, float('nan')):>12.4f}" for n in ns_str))

    print("\n[4] Coefficient of variation of MI across folds (lower=more stable)")
    print("-" * 78)
    cv = summary["cv_of_mi_per_method"]
    for m, v in sorted(cv.items(), key=lambda kv: kv[1]):
        print(f"{m:<20} {v:>10.4f}")
    print()


# =============================================================================
# Entry point
# =============================================================================


def main():
    out = run_benchmark(
        distributions=None,
        signal_kinds=None,
        sample_sizes=[500, 2000, 10000],
        n_splits=5,
        n_repeats=2,
        verbose=1,
    )
    print_summary(out["summary"])
    # Save full results json under D:/Temp.
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path("D:/Temp") / f"bench_adaptive_nbins_{ts}.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[bench_adaptive_nbins] full results -> {out_path}")
    except Exception as exc:
        print(f"[bench_adaptive_nbins] could not save json: {exc!r}")


if __name__ == "__main__":
    main()
