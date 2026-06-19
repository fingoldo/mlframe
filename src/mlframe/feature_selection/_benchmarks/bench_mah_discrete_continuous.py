"""MAH/SCI discrete-continuous focused bench (2026-05-29 Wave 7).

The Marx-Yang-van Leeuwen 2021 SDM paper's headline claim is that MAH/SCI
DOMINATES on discrete-continuous mixtures (Tables 2-4 of the paper). My
generic 8-distribution mega-bench tested mostly continuous-continuous which
is NOT the paper's target regime; MAH there under-estimated by ~10x because
greedy SCI-minimisation prefers fewer bins under continuous-continuous
joints.

This focused bench reproduces the paper's actual claim setup:
  * y in {0, 1} OR {0..K-1} (discrete) -- the regime the paper targets.
  * x continuous (varied distributions).
  * True MI estimated via large-N Mixed-KSG reference (N=100k).
  * Compare: MAH/SCI, plug-in (FD, MDLP, OptimalJoint), Mixed-KSG.

The expected pattern (per paper): MAH should approach Mixed-KSG accuracy
while staying calibrated under independence -- a property that plug-in
methods lose at high bin counts (Miller-Madow noise floor).

Run::

    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_mah_discrete_continuous
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from mlframe.feature_selection.filters._adaptive_nbins import (
    per_feature_edges, _plug_in_mi,
)
from mlframe.feature_selection.filters._ksg import mixed_ksg_mi
from mlframe.feature_selection.filters._mah import mah_mi


# =============================================================================
# Synthetic discrete-continuous y suite
# =============================================================================


def _draw_x(distribution: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if distribution == "gaussian":
        return rng.standard_normal(n)
    if distribution == "lognormal":
        return rng.lognormal(size=n)
    if distribution == "heavy_tail_t":
        return rng.standard_t(df=3, size=n)
    if distribution == "exponential":
        return rng.exponential(scale=1.0, size=n)
    if distribution == "bimodal":
        flag = rng.integers(0, 2, n).astype(np.float64)
        return flag * rng.normal(loc=-2, scale=0.5, size=n) + \
               (1 - flag) * rng.normal(loc=2, scale=0.5, size=n)
    raise ValueError(distribution)


def _draw_discrete_y(x: np.ndarray, signal: str, K: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Generate K-class discrete y conditioned on continuous x.

    Signals:
      * 'no_signal': y is independent of x.
      * 'monotone': y is a K-class binning of x + noise.
      * 'threshold': y is the indicator of x exceeding K-1 quantiles.
      * 'xor': y is the K-class threshold-grid PARITY (XOR generalisation).
    """
    n = x.size
    if signal == "no_signal":
        return rng.integers(0, K, n).astype(np.int64)
    if signal == "monotone":
        z = x + rng.standard_normal(n) * 0.3
        ranks = np.argsort(np.argsort(z))
        return (ranks * K // n).astype(np.int64)
    if signal == "threshold":
        cuts = np.quantile(x, np.linspace(0, 1, K + 1)[1:-1])
        return np.searchsorted(cuts, x).astype(np.int64)
    if signal == "xor":
        cuts = np.quantile(x, np.linspace(0, 1, 5)[1:-1])
        seg = np.searchsorted(cuts, x)
        return (seg % K).astype(np.int64)
    raise ValueError(signal)


# =============================================================================
# Estimators
# =============================================================================


def _est_plug_in_fd(x, y, miller_madow=True):
    edges = per_feature_edges(x.reshape(-1, 1), method="freedman_diaconis")[0]
    xb = np.searchsorted(edges, x, side="right").astype(np.int64)
    return _plug_in_mi(xb, y.astype(np.int64), miller_madow=miller_madow)


def _est_plug_in_mdlp(x, y):
    edges = per_feature_edges(x.reshape(-1, 1), y=y, method="mdlp")[0]
    xb = np.searchsorted(edges, x, side="right").astype(np.int64)
    return _plug_in_mi(xb, y.astype(np.int64), miller_madow=True)


def _est_plug_in_optimal_joint(x, y):
    edges = per_feature_edges(x.reshape(-1, 1), y=y, method="optimal_joint")[0]
    xb = np.searchsorted(edges, x, side="right").astype(np.int64)
    return _plug_in_mi(xb, y.astype(np.int64), miller_madow=True)


def _est_mixed_ksg(x, y):
    return mixed_ksg_mi(x, y.astype(np.float64), k=5)


def _est_mah(x, y):
    return mah_mi(x, y.astype(np.float64), initial_k=16)


ESTIMATORS: Dict[str, Callable] = {
    "plug_in_fd": _est_plug_in_fd,
    "plug_in_mdlp": _est_plug_in_mdlp,
    "plug_in_optimal_joint": _est_plug_in_optimal_joint,
    "mixed_ksg": _est_mixed_ksg,
    "mah_sci": _est_mah,
}


# =============================================================================
# Bench
# =============================================================================


@dataclass
class Row:
    estimator: str
    distribution: str
    signal: str
    K: int
    n: int
    mi_val: float
    truth: float
    runtime_ms: float


def _truth(distribution: str, signal: str, K: int, n_truth: int = 100_000,
            seed: int = 42) -> float:
    """Large-N Mixed-KSG reference for the (distribution, signal, K) cell."""
    rng = np.random.default_rng(int(seed))
    x = _draw_x(distribution, n_truth, rng)
    y = _draw_discrete_y(x, signal, K, rng)
    return float(mixed_ksg_mi(x, y.astype(np.float64), k=5))


def run_bench(N: int = 5000, n_repeats: int = 2,
               distributions=None, signals=None, K_values=None,
               estimators=None, seed: int = 0, verbose: int = 1):
    distributions = distributions or ["gaussian", "lognormal", "heavy_tail_t",
                                       "exponential", "bimodal"]
    signals = signals or ["no_signal", "monotone", "threshold", "xor"]
    K_values = K_values or [2, 3, 5, 10]
    estimators = estimators or ESTIMATORS
    rng = np.random.default_rng(int(seed))

    truths: Dict = {}
    results: List[Row] = []
    total = (len(distributions) * len(signals) * len(K_values)
              * n_repeats * len(estimators))
    if verbose:
        print(f"[mah-disc] {total} (est, dist, sig, K, rep) cells; N per cell={N}")

    counter = 0
    for dist in distributions:
        for sig in signals:
            for K in K_values:
                tkey = (dist, sig, K)
                if tkey not in truths:
                    truths[tkey] = _truth(dist, sig, K)
                t = truths[tkey]
                for rep in range(n_repeats):
                    sub_rng = np.random.default_rng(int(rng.integers(0, 2**30)))
                    x = _draw_x(dist, N, sub_rng)
                    y = _draw_discrete_y(x, sig, K, sub_rng)
                    for name, fn in estimators.items():
                        t0 = time.perf_counter()
                        try:
                            mi = float(fn(x, y))
                        except Exception as exc:
                            mi = float("nan")
                            if verbose >= 2:
                                print(f"  FAIL {name}: {exc}")
                        rt = (time.perf_counter() - t0) * 1000.0
                        results.append(Row(
                            estimator=name, distribution=dist, signal=sig,
                            K=K, n=N, mi_val=mi, truth=t, runtime_ms=rt,
                        ))
                        counter += 1
                if verbose:
                    print(f"  done dist={dist:<12} sig={sig:<10} K={K} "
                          f"({counter}/{total})")
    return {"results": [asdict(r) for r in results]}


def print_summary(results_dicts):
    print("\n" + "=" * 110)
    print("MAH/SCI discrete-continuous bench -- paper's headline regime")
    print("=" * 110)
    from collections import defaultdict
    by_est = defaultdict(list)
    by_est_sig = defaultdict(lambda: defaultdict(list))
    for r in results_dicts:
        by_est[r["estimator"]].append(r)
        by_est_sig[r["estimator"]][r["signal"]].append(r)

    # Overall.
    print(f"\n{'estimator':<24} {'mean_abs_err':>14} {'mean_rt_ms':>12} "
          f"{'n_cells':>8}")
    print("-" * 110)
    for est, rs in sorted(by_est.items(),
                            key=lambda kv: np.nanmean([abs(r["mi_val"] - r["truth"])
                                                         for r in kv[1]])):
        errs = [abs(r["mi_val"] - r["truth"]) for r in rs
                 if np.isfinite(r["mi_val"])]
        rts = [r["runtime_ms"] for r in rs]
        print(f"{est:<24} {np.nanmean(errs):>14.4f} {np.nanmean(rts):>12.2f} "
              f"{len(rs):>8}")
    # Per-signal.
    signals = sorted({r["signal"] for r in results_dicts})
    print(f"\nPer-signal mean |error| vs truth:")
    print(f"{'estimator':<24}" + ''.join(f'{s:>12}' for s in signals))
    print("-" * 110)
    for est in by_est:
        row = []
        for s in signals:
            sub = by_est_sig[est].get(s, [])
            if not sub:
                row.append(float("nan"))
            else:
                row.append(np.nanmean([abs(r["mi_val"] - r["truth"]) for r in sub
                                        if np.isfinite(r["mi_val"])]))
        print(f"{est:<24}" + ''.join(f'{v:>12.4f}' for v in row))
    # Per-K.
    Ks = sorted({r["K"] for r in results_dicts})
    print(f"\nPer-K (number of y classes) mean |error|:")
    print(f"{'estimator':<24}" + ''.join(f'{("K=" + str(k)):>10}' for k in Ks))
    print("-" * 110)
    for est in by_est:
        row = []
        for k in Ks:
            sub = [r for r in by_est[est] if r["K"] == k]
            if not sub:
                row.append(float("nan"))
            else:
                row.append(np.nanmean([abs(r["mi_val"] - r["truth"]) for r in sub
                                        if np.isfinite(r["mi_val"])]))
        print(f"{est:<24}" + ''.join(f'{v:>10.4f}' for v in row))


def main():
    out = run_bench(N=5000, n_repeats=2, verbose=1)
    print_summary(out["results"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = Path("D:/Temp") / f"bench_mah_discrete_continuous_{ts}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[mah-disc] -> {path}")
    except Exception as exc:
        print(f"[mah-disc] save failed: {exc!r}")


if __name__ == "__main__":
    main()
