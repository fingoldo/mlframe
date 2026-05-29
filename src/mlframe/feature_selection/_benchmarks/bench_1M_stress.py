"""1M-row stress bench of selected default MI estimators (2026-05-29).

Production sanity check: makes sure the recommended defaults don't blow up on
realistic dataset sizes (1M rows x 1 feature pair). Reports wall-time, peak
RAM, and accuracy vs Monte-Carlo truth.

Tested estimators (mega-bench v3 honest leaders):
  * plug_in (FD nbins_strategy + Miller-Madow) -- speed champion
  * Mixed-KSG -- honest non-aggregator non-neural
  * KSG-LNC -- with low-entropy-skip; falls back to Mixed-KSG on binary y
  * GENIE aggregator -- honest leader on noisy continuous
  * Median aggregator -- zero-cost robust default
  * fastMI MISE -- copula FFT-KDE
  * MIST calibrated -- pre-trained transformer (binary y calibration table)
  * InfoNet -- pre-trained transformer
  * MINE bootstrap -- neural DV with small-N bootstrap

Reported per (estimator, signal_kind, distribution):
  * mi_val (estimate)
  * truth (large-N MC reference)
  * abs_error
  * runtime_seconds
  * peak_ram_mb

Run::

    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_1M_stress
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
import tracemalloc
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np


# =============================================================================
# Synthetic data
# =============================================================================


def _gen_data(distribution: str, signal_kind: str, n: int, seed: int):
    rng = np.random.default_rng(int(seed))
    if distribution == "gaussian":
        x = rng.standard_normal(n)
    elif distribution == "lognormal":
        x = rng.lognormal(size=n)
    elif distribution == "heavy_tail_t":
        x = rng.standard_t(df=3, size=n)
    elif distribution == "discrete_low_card":
        x = rng.integers(0, 8, n).astype(np.float64)
    else:
        raise ValueError(distribution)
    if signal_kind == "no_signal":
        y = rng.integers(0, 2, n).astype(np.float64)
    elif signal_kind == "linear":
        z = x + rng.standard_normal(n) * 0.3
        y = (z > np.median(z)).astype(np.float64)
    elif signal_kind == "threshold":
        thr = np.quantile(x, 0.7)
        y = (x > thr).astype(np.float64)
    elif signal_kind == "xor":
        thr_lo = np.quantile(x, 0.33)
        thr_hi = np.quantile(x, 0.67)
        y = ((x > thr_lo) & (x < thr_hi)).astype(np.float64)
    else:
        raise ValueError(signal_kind)
    return x.astype(np.float64), y


# =============================================================================
# Estimators (production defaults)
# =============================================================================


def _est_plug_in(x, y):
    from mlframe.feature_selection.filters._adaptive_nbins import (
        edges_freedman_diaconis, _plug_in_mi,
    )
    e = edges_freedman_diaconis(x)
    xb = np.searchsorted(e, x, side="right").astype(np.int64)
    return _plug_in_mi(xb, y.astype(np.int64), miller_madow=True)


def _est_mixed_ksg(x, y):
    from mlframe.feature_selection.filters._ksg import mixed_ksg_mi
    return mixed_ksg_mi(x, y, k=5)


def _est_ksg_lnc(x, y):
    from mlframe.feature_selection.filters._ksg import ksg_lnc_mi
    return ksg_lnc_mi(x, y, k=5)


def _est_genie(x, y):
    from mlframe.feature_selection.filters._mi_aggregator import genie_mi_panel
    from mlframe.feature_selection.filters._ksg import mixed_ksg_mi
    from mlframe.feature_selection.filters._adaptive_nbins import (
        edges_freedman_diaconis, edges_qs, _plug_in_mi,
    )
    def _fd(a, b):
        e = edges_freedman_diaconis(a)
        bb = np.searchsorted(e, a, side="right")
        return _plug_in_mi(bb, np.asarray(b).astype(np.int64), miller_madow=True)
    def _qs(a, b):
        e = edges_qs(a)
        bb = np.searchsorted(e, a, side="right")
        return _plug_in_mi(bb, np.asarray(b).astype(np.int64), miller_madow=True)
    def _ksg(a, b):
        return mixed_ksg_mi(a, np.asarray(b).astype(np.float64), k=5)
    return genie_mi_panel(x, y, {"fd": _fd, "qs": _qs, "ksg": _ksg})


def _est_fastmi(x, y):
    from mlframe.feature_selection.filters._fastmi import fastmi
    return fastmi(x, y, bandwidth="mise")


def _est_mist(x, y):
    from mlframe.feature_selection.filters._neural_mi import mist_mi
    return mist_mi(x, y, calibrated=True, device="auto")


def _est_infonet(x, y):
    from mlframe.feature_selection.filters._neural_mi import infonet_mi
    return infonet_mi(x, y, device="auto")


def _est_mine(x, y):
    from mlframe.feature_selection.filters._neural_mi import mine_mi
    return mine_mi(x, y, n_epochs=400, bootstrap_to_n=0, device="auto")


ESTIMATORS: Dict[str, Callable] = {
    "plug_in_fd_mm": _est_plug_in,
    "mixed_ksg": _est_mixed_ksg,
    "ksg_lnc": _est_ksg_lnc,
    "genie_panel": _est_genie,
    "fastmi_mise": _est_fastmi,
    "mist_calibrated": _est_mist,
    "infonet": _est_infonet,
    "mine": _est_mine,
}


# =============================================================================
# Bench loop
# =============================================================================


@dataclass
class StressFold:
    estimator: str
    distribution: str
    signal_kind: str
    n: int
    mi_val: float
    truth: float
    runtime_s: float
    peak_ram_mb: float
    error_msg: str = ""


def _measure_truth(distribution: str, signal_kind: str, n_truth: int = 200_000,
                   seed: int = 0) -> float:
    """MC truth via Mixed-KSG on N=200k."""
    from mlframe.feature_selection.filters._ksg import mixed_ksg_mi
    x, y = _gen_data(distribution, signal_kind, n_truth, seed=seed)
    return float(mixed_ksg_mi(x, y, k=5))


def run_stress_bench(N: int = 1_000_000,
                      distributions=None, signal_kinds=None,
                      estimators=None, seed: int = 0,
                      verbose: int = 1) -> Dict:
    distributions = distributions or ["gaussian", "lognormal", "heavy_tail_t",
                                       "discrete_low_card"]
    signal_kinds = signal_kinds or ["no_signal", "linear", "threshold", "xor"]
    estimators = estimators or ESTIMATORS
    rng = np.random.default_rng(int(seed))

    results: List[StressFold] = []
    truths_cache: Dict = {}
    total = len(distributions) * len(signal_kinds) * len(estimators)
    if verbose:
        print(f"[1M-stress] N={N:,}; {total} (dist, signal, estimator) cells")

    counter = 0
    for dist in distributions:
        for sig in signal_kinds:
            # Compute truth ONCE per (dist, signal).
            tkey = (dist, sig)
            if tkey not in truths_cache:
                truths_cache[tkey] = _measure_truth(dist, sig, seed=42)
            truth = truths_cache[tkey]
            x, y = _gen_data(dist, sig, N, seed=int(rng.integers(0, 2**30)))
            for est_name, est_fn in estimators.items():
                counter += 1
                gc.collect()
                tracemalloc.start()
                t0 = time.perf_counter()
                err_msg = ""
                mi = float("nan")
                try:
                    mi = float(est_fn(x, y))
                except Exception as exc:
                    err_msg = f"{type(exc).__name__}: {exc}"
                runtime = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                results.append(StressFold(
                    estimator=est_name, distribution=dist, signal_kind=sig,
                    n=N, mi_val=mi, truth=truth, runtime_s=runtime,
                    peak_ram_mb=peak / 1024 / 1024,
                    error_msg=err_msg,
                ))
                if verbose:
                    status = err_msg or f"MI={mi:.4f}"
                    print(f"  [{counter}/{total}] {est_name:<18} {dist:<18} "
                          f"{sig:<10}: {status} truth={truth:.4f} "
                          f"rt={runtime:.1f}s peak={peak/1024/1024:.0f}MB")
                gc.collect()
            del x, y
    return {"results": [asdict(r) for r in results]}


def print_summary(results_dicts: List[Dict]) -> None:
    print("\n" + "=" * 110)
    print(f"1M-ROW STRESS BENCH — Production sanity check")
    print("=" * 110)
    # Per-estimator aggregates.
    by_est: Dict[str, List[Dict]] = {}
    for r in results_dicts:
        by_est.setdefault(r["estimator"], []).append(r)
    print(f"{'estimator':<20} {'mean_abs_err':>14} {'mean_rt_s':>11} "
          f"{'max_rt_s':>10} {'peak_ram_MB':>13} {'n_fail':>8}")
    print("-" * 110)
    for est, rs in sorted(by_est.items(),
                            key=lambda kv: np.mean([abs(r["mi_val"] - r["truth"])
                                                     for r in kv[1]
                                                     if not r["error_msg"]
                                                     and np.isfinite(r["mi_val"])])):
        ok = [r for r in rs if not r["error_msg"] and np.isfinite(r["mi_val"])]
        if not ok:
            print(f"{est:<20} {'-':>14} {'-':>11} {'-':>10} {'-':>13} {len(rs):>8}")
            continue
        abs_errs = [abs(r["mi_val"] - r["truth"]) for r in ok]
        rts = [r["runtime_s"] for r in ok]
        peaks = [r["peak_ram_mb"] for r in ok]
        n_fail = len(rs) - len(ok)
        print(f"{est:<20} {np.mean(abs_errs):>14.4f} {np.mean(rts):>11.2f} "
              f"{max(rts):>10.2f} {max(peaks):>13.0f} {n_fail:>8}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--quick", action="store_true",
                    help="run on N=100k instead of 1M (development check)")
    args = ap.parse_args()
    N = 100_000 if args.quick else args.n
    out = run_stress_bench(N=N, verbose=1)
    print_summary(out["results"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = Path("D:/Temp") / f"bench_1M_stress_{ts}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[1M-stress] -> {path}")
    except Exception as exc:
        print(f"[1M-stress] save failed: {exc!r}")


if __name__ == "__main__":
    main()
