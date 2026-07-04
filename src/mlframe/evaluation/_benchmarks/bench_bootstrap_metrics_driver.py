"""Bench harness for the bootstrap_metrics DRIVER (agentC, 2026-07).

Mirrors the honest-diagnostics binary-classification bootstrap-CI call:
stratified resample, metric_fns={brier, log_loss} (slice-based), metric_fns_idx=
{roc_auc} (index-aware), per_row_fns + jackknife_fns for the fast BCa jackknife.

Measures the FULL bootstrap_metrics wall (best-of-N median) at n in {200k, 1M}.
Run:  python -m mlframe.evaluation._benchmarks.bench_bootstrap_metrics_driver
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from statistics import median

import numpy as np

from mlframe.evaluation.bootstrap import bootstrap_metrics, _jackknife_auc
from mlframe.metrics.core import (
    fast_brier_score_loss as _fast_brier,
    fast_log_loss as _fast_ll,
    make_bootstrap_auc_resampler,
)


def build_inputs(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < 0.3).astype(np.int64)
    # separable-ish scores so AUC ~0.8, both classes present
    p = np.clip(0.2 + 0.5 * y_true + rng.normal(0, 0.25, n), 1e-4, 1 - 1e-4)
    y_f64 = np.ascontiguousarray(y_true, dtype=np.float64)
    p_f64 = np.ascontiguousarray(p, dtype=np.float64)
    return y_true, y_f64, p_f64


def build_kwargs(y_true, y_f64, p_f64):
    def _brier(yy, pp):
        return float(_fast_brier(yy, pp))

    def _ll(yy, pp):
        return float(_fast_ll(yy, pp))

    def _ll_per_row(yy, pp):
        _eps = np.finfo(np.asarray(pp).dtype).eps
        _pc = np.clip(pp, _eps, 1.0 - _eps)
        return np.where(np.asarray(yy) == 1, -np.log(_pc), -np.log(1.0 - _pc))

    def _brier_per_row(yy, pp):
        _d = np.asarray(pp, dtype=np.float64) - np.asarray(yy, dtype=np.float64)
        return _d * _d

    metric_fns = {"brier": _brier, "log_loss": _ll}
    per_row_fns = {"log_loss": (_ll_per_row, True, None), "brier": (_brier_per_row, False, None)}
    jackknife_fns = {"roc_auc": lambda yy, ss: _jackknife_auc(yy, ss)}
    metric_fns_idx = {"roc_auc": make_bootstrap_auc_resampler(y_f64, p_f64)}
    return dict(
        metric_fns=metric_fns, metric_fns_idx=metric_fns_idx,
        per_row_fns=per_row_fns, jackknife_fns=jackknife_fns,
    )


def run_once(y_true, y_f64, p_f64, n_bootstrap=1000, seed=12345):
    kw = build_kwargs(y_true, y_f64, p_f64)
    return bootstrap_metrics(
        y_f64, p_f64, kw["metric_fns"], n_bootstrap=n_bootstrap, alpha=0.05,
        stratify=y_true, random_state=seed,
        metric_fns_idx=kw["metric_fns_idx"], per_row_fns=kw["per_row_fns"],
        jackknife_fns=kw["jackknife_fns"],
    )


def bench(n, reps=5, n_bootstrap=1000):
    y_true, y_f64, p_f64 = build_inputs(n)
    run_once(y_true, y_f64, p_f64, n_bootstrap=min(50, n_bootstrap))  # warm numba
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        res = run_once(y_true, y_f64, p_f64, n_bootstrap=n_bootstrap)
        times.append(time.perf_counter() - t0)
    return median(times), res


if __name__ == "__main__":
    do_profile = "--profile" in sys.argv
    for n in (200_000, 1_000_000):
        med, res = bench(n, reps=5)
        keys = {k: (round(v.get("point", float("nan")), 6),
                    round(v.get("lo", float("nan")), 6),
                    round(v.get("hi", float("nan")), 6)) for k, v in res.items()}
        print(f"n={n:>9}  median wall={med:.3f}s  {keys}")
    if do_profile:
        y_true, y_f64, p_f64 = build_inputs(1_000_000)
        run_once(y_true, y_f64, p_f64, n_bootstrap=50)
        pr = cProfile.Profile()
        pr.enable()
        run_once(y_true, y_f64, p_f64, n_bootstrap=1000)
        pr.disable()
        st = pstats.Stats(pr).sort_stats("tottime")
        st.print_stats(20)
