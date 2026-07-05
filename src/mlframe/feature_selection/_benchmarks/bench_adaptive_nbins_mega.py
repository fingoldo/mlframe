"""Mega-bench: ALL methods (binning, KSG, neural, copula, aggregator) head-to-head.

Extends the WAVE 1 A/B bench with the new method families:

  * **QS** (Gupta 2021)               - cheap noise-floor specialist.
  * **Mixed-KSG** (Gao 2017)          - SOTA non-neural baseline (Czyz 2023).
  * **KSG-LNC** (Gao 2015)            - experimental local-PCA correction.
  * **MINE** (Belghazi 2018)          - PyTorch+CUDA Donsker-Varadhan.
  * **fastMI** (Purkayastha-Song 2024) - copula+FFT-KDE.
  * **median(FD, QS, KSG)** aggregator.
  * **GENIE(FD, QS, KSG)** aggregator.

Designed to run AFTER the WAVE 1 A/B bench produces its delta table; this
bench answers "given the WAVE 1 fixes are in, which method is the new best?".

Sample-size grid is reduced ({500, 2000}) because MINE adds ~2s per fold and
GPU compilation amortises; n=10000 included only for n_repeats=1 to bound cost.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from mlframe.feature_selection.filters._adaptive_nbins import (
    per_feature_edges,
    _plug_in_mi,
    edges_freedman_diaconis,
    edges_qs,
)
from mlframe.feature_selection.filters._ksg import mixed_ksg_mi, ksg_lnc_mi
from mlframe.feature_selection.filters._mi_aggregator import (
    median_mi_panel, genie_mi_panel,
)
from mlframe.feature_selection._benchmarks.bench_adaptive_nbins import (
    _draw_distribution, _draw_signal, _make_folds,
    _legacy_quantile_edges, _bin_with_edges,
)

# =============================================================================
# Score wrappers (uniform API: x_train, y_train, x_val, y_val -> mi_val)
# =============================================================================


def _score_binning(x_tr, y_tr, x_va, y_va, method: str, miller_madow: bool, **method_kwargs) -> float:
    """Binning-style scorer: fit edges on train, bin val, plug-in MI."""
    X_tr = x_tr.reshape(-1, 1)
    if method == "quantile10":
        edges = _legacy_quantile_edges(x_tr, n_bins=10)
    else:
        edges_list = per_feature_edges(X_tr, y_tr, method=method, **method_kwargs)
        edges = edges_list[0]
    val_b = _bin_with_edges(x_va, edges)
    return _plug_in_mi(val_b, y_va.astype(np.int64), miller_madow=miller_madow), int(edges.size + 1)


def _score_mixed_ksg(x_tr, y_tr, x_va, y_va, k: int = 5) -> float:
    """Mixed-KSG: train fold ignored; scoring on val (x, y) directly."""
    return mixed_ksg_mi(x_va, y_va.astype(np.float64), k=k), 0


def _score_ksg_lnc(x_tr, y_tr, x_va, y_va, k: int = 5, alpha: float = 0.65) -> float:
    return ksg_lnc_mi(x_va, y_va.astype(np.float64), k=k, alpha=alpha), 0


def _score_mine(x_tr, y_tr, x_va, y_va, n_epochs: int = 200, seed: int = 0) -> float:
    from mlframe.feature_selection.filters._neural_mi import mine_mi

    return mine_mi(x_va, y_va.astype(np.float64), n_epochs=n_epochs, seed=seed, device="auto"), 0


def _score_infonet(x_tr, y_tr, x_va, y_va, seed: int = 0) -> float:
    from mlframe.feature_selection.filters._neural_mi import infonet_mi
    return infonet_mi(x_va, y_va.astype(np.float64), seed=seed, device="auto"), 0


def _score_mist(x_tr, y_tr, x_va, y_va) -> float:
    from mlframe.feature_selection.filters._neural_mi import mist_mi
    return mist_mi(x_va, y_va.astype(np.float64), device="auto"), 0


def _score_fastmi_silv(x_tr, y_tr, x_va, y_va, grid_size: int = 128) -> float:
    from mlframe.feature_selection.filters._fastmi import fastmi

    return fastmi(x_va, y_va.astype(np.float64), grid_size=grid_size, bandwidth="silverman"), 0


def _score_fastmi_mise(x_tr, y_tr, x_va, y_va, grid_size: int = 128) -> float:
    from mlframe.feature_selection.filters._fastmi import fastmi

    return fastmi(x_va, y_va.astype(np.float64), grid_size=grid_size, bandwidth="mise"), 0


def _score_median_panel(x_tr, y_tr, x_va, y_va) -> float:
    def _fd(a, b):
        e = edges_freedman_diaconis(a)
        bb = np.searchsorted(e, a.astype(np.float64), side="right")
        return _plug_in_mi(bb, b.astype(np.int64), miller_madow=True)
    def _qs(a, b):
        e = edges_qs(a)
        bb = np.searchsorted(e, a.astype(np.float64), side="right")
        return _plug_in_mi(bb, b.astype(np.int64), miller_madow=True)
    def _ksg(a, b):
        return mixed_ksg_mi(a, b.astype(np.float64), k=5)
    return median_mi_panel(x_va, y_va, {"fd": _fd, "qs": _qs, "ksg": _ksg}), 0


def _score_genie_panel(x_tr, y_tr, x_va, y_va) -> float:
    def _fd(a, b):
        e = edges_freedman_diaconis(a)
        bb = np.searchsorted(e, a.astype(np.float64), side="right")
        return _plug_in_mi(bb, b.astype(np.int64), miller_madow=True)
    def _qs(a, b):
        e = edges_qs(a)
        bb = np.searchsorted(e, a.astype(np.float64), side="right")
        return _plug_in_mi(bb, b.astype(np.int64), miller_madow=True)
    def _ksg(a, b):
        return mixed_ksg_mi(a, b.astype(np.float64), k=5)
    return genie_mi_panel(x_va, y_va, {"fd": _fd, "qs": _qs, "ksg": _ksg}), 0


# WAVE 1 fixes applied as "the production defaults" for binning methods.
BINNING_KWARGS = {
    "quantile10": {},
    "sturges": {},
    "freedman_diaconis": {},
    "qs": {},
    "knuth": {"knuth_edge_type": "quantile", "knuth_m_max_cap": 64},
    "blocks": {"bb_edge_placement": "midpoint", "p0": 0.10, "bb_subsample_threshold": 1000},
    "fayyad_irani": {"mdlp_backend": "njit", "mdlp_scaled_min_split": True},
    "optimal_joint": {},
}

# Mapping method -> (label, scoring callable). All binning methods use the
# same _score_binning; KSG / MINE / fastMI / aggregator each have their own.
def _make_scorers() -> Dict[str, Callable]:
    scorers: Dict[str, Callable] = {}
    for method in ["quantile10", "sturges", "freedman_diaconis", "qs", "knuth", "blocks", "fayyad_irani", "optimal_joint"]:
        kw = BINNING_KWARGS[method]
        def make(m=method, mkw=kw):
            def _score(x_tr, y_tr, x_va, y_va):
                return _score_binning(x_tr, y_tr, x_va, y_va, m, miller_madow=True, **mkw)
            return _score
        scorers[f"{method}_w1fixes"] = make()
        # Also include the no-MM legacy for reference on the same row.
        if method in ("freedman_diaconis", "quantile10"):
            def make_nomm(m=method, mkw=kw):
                def _score(x_tr, y_tr, x_va, y_va):
                    return _score_binning(x_tr, y_tr, x_va, y_va, m, miller_madow=False, **mkw)
                return _score
            scorers[f"{method}_legacy"] = make_nomm()
    scorers["mixed_ksg"] = lambda xt, yt, xv, yv: _score_mixed_ksg(xt, yt, xv, yv)
    scorers["ksg_lnc"] = lambda xt, yt, xv, yv: _score_ksg_lnc(xt, yt, xv, yv)
    scorers["mine"] = lambda xt, yt, xv, yv: _score_mine(xt, yt, xv, yv, n_epochs=200)
    scorers["infonet"] = lambda xt, yt, xv, yv: _score_infonet(xt, yt, xv, yv)
    scorers["mist"] = lambda xt, yt, xv, yv: _score_mist(xt, yt, xv, yv)
    scorers["fastmi_silv"] = lambda xt, yt, xv, yv: _score_fastmi_silv(xt, yt, xv, yv)
    scorers["fastmi_mise"] = lambda xt, yt, xv, yv: _score_fastmi_mise(xt, yt, xv, yv)
    scorers["median_panel"] = lambda xt, yt, xv, yv: _score_median_panel(xt, yt, xv, yv)
    scorers["genie_panel"] = lambda xt, yt, xv, yv: _score_genie_panel(xt, yt, xv, yv)
    return scorers


# =============================================================================
# Bench driver
# =============================================================================


@dataclass
class MegaFoldResult:
    method: str
    distribution: str
    signal_kind: str
    n: int
    fold_idx: int
    mi_val: float
    nbins: int
    runtime_ms: float


def run_mega_bench(
    distributions: Optional[List[str]] = None,
    signal_kinds: Optional[List[str]] = None,
    sample_sizes: Optional[List[int]] = None,
    n_splits: int = 3,
    n_repeats: int = 1,
    methods_subset: Optional[List[str]] = None,
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
    sample_sizes = sample_sizes or [500, 2000]
    scorers = _make_scorers()
    if methods_subset is not None:
        scorers = {k: v for k, v in scorers.items() if k in methods_subset}

    rng_master = np.random.default_rng(random_state)
    results: List[MegaFoldResult] = []
    total = len(distributions) * len(signal_kinds) * len(sample_sizes) * n_repeats * n_splits * len(scorers)
    if verbose:
        print(f"[mega] {total} fold-method evaluations across {len(scorers)} methods")

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
                        xt, yt = x[tr_idx], y[tr_idx]
                        xv, yv = x[va_idx], y[va_idx]
                        for mname, scorer in scorers.items():
                            try:
                                t0 = time.perf_counter()
                                out = scorer(xt, yt, xv, yv)
                                mi_val, nbins = out if isinstance(out, tuple) else (out, 0)
                                t_ms = (time.perf_counter() - t0) * 1000.0
                            except Exception as exc:
                                if verbose >= 2:
                                    print(f"  [FAIL] {mname} ({dist},{sig},n={n},f={fold_idx}): {exc!r}")
                                continue
                            results.append(MegaFoldResult(
                                method=mname, distribution=dist, signal_kind=sig,
                                n=n, fold_idx=fold_idx, mi_val=float(mi_val),
                                nbins=int(nbins), runtime_ms=float(t_ms),
                            ))
                            counter += 1
                if verbose:
                    print(f"  done: dist={dist:15s} signal={sig:12s} ({counter}/{total})")

    return {"results": [asdict(r) for r in results], "summary": _summarise(results)}


def _summarise(results: List[MegaFoldResult]) -> Dict:
    by_m: Dict[str, List[MegaFoldResult]] = {}
    for r in results:
        by_m.setdefault(r.method, []).append(r)
    per_m: Dict[str, Dict] = {}
    no_sig: Dict[str, float] = {}
    for m, rs in by_m.items():
        mi = np.array([r.mi_val for r in rs])
        rt = np.array([r.runtime_ms for r in rs])
        per_m[m] = {
            "mi_mean": float(mi.mean()), "mi_median": float(np.median(mi)),
            "mi_q25": float(np.percentile(mi, 25)), "mi_q75": float(np.percentile(mi, 75)),
            "rt_ms_mean": float(rt.mean()), "rt_ms_med": float(np.median(rt)),
            "n_folds": len(rs),
        }
        ns = [r.mi_val for r in rs if r.signal_kind == "no_signal"]
        if ns:
            no_sig[m] = float(np.mean(ns))
    # Win rate.
    by_task: Dict[Tuple, Dict[str, float]] = {}
    for r in results:
        key = (r.distribution, r.signal_kind, r.n, r.fold_idx)
        by_task.setdefault(key, {})[r.method] = r.mi_val
    wins: Dict[str, int] = {m: 0 for m in by_m}
    total_tasks = 0
    for _, mv in by_task.items():
        if not mv:
            continue
        winner = max(mv, key=mv.get)
        wins[winner] = wins.get(winner, 0) + 1
        total_tasks += 1
    win_rate = {m: (wins.get(m, 0) / max(1, total_tasks)) for m in by_m}
    return {"per_method": per_m, "no_signal_mi": no_sig, "win_rate": win_rate}


def print_mega_summary(summary: Dict) -> None:
    pm = summary["per_method"]
    ns = summary["no_signal_mi"]
    wr = summary["win_rate"]
    print("\n" + "=" * 96)
    print("MEGA BENCH: all MI estimators head-to-head (WAVE 1 fixes applied on binning)")
    print("=" * 96)
    print(f"{'method':<28} {'MI_mean':>9} {'MI_med':>8} {'no_sig':>8} " f"{'rt_ms':>9} {'win_rate':>9}")
    print("-" * 96)
    for m, d in sorted(pm.items(), key=lambda kv: -kv[1]["mi_mean"]):
        print(f"{m:<28} {d['mi_mean']:>9.4f} {d['mi_median']:>8.4f} " f"{ns.get(m, float('nan')):>8.4f} {d['rt_ms_mean']:>9.3f} " f"{wr.get(m, 0.0):>9.2%}")
    print()


def main():
    out = run_mega_bench(
        sample_sizes=[500, 2000],
        n_splits=3, n_repeats=1, verbose=1,
    )
    print_mega_summary(out["summary"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = Path("D:/Temp") / f"bench_adaptive_nbins_mega_{ts}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[mega] -> {path}")
    except Exception as exc:
        print(f"[mega] save failed: {exc!r}")


if __name__ == "__main__":
    main()
