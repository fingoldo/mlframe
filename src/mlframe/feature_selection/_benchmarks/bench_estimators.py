"""Benchmark accuracy + speed of MI estimators / mRMR variants.

Production-realistic ``n=200_000, p=50, n_informative=15`` -- on this
scale numba JIT cache, GPU memory pools, and joblib worker spawn cost
are all amortised. **All paths get a warmup pass before timing**;
reported wall-times reflect steady-state cost, not first-call compile.

Compares:
* Plug-in (legacy, discretized)
* MRMR + Adaptive Besag-Clifford permutation (parallelism="bc")
* MRMR + Stability Selection (Meinshausen-Buhlmann bootstrap)
* MRMR + GroupAware (correlation pre-clustering)
* KSG top-K (no significance test)
* KSG + permutation significance test

Recovery rate (precision against the known set of informative features
from ``make_classification``) is the accuracy metric; wall-time per
fit is the speed metric.

Run::

    python -m mlframe.feature_selection._benchmarks.bench_estimators
    python -m mlframe.feature_selection._benchmarks.bench_estimators --n 50000 --p 100
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger("bench_estimators")


def _build(n: int, p: int, n_inf: int, seed: int):
    X, y = make_classification(
        n_samples=n,
        n_features=p,
        n_informative=n_inf,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=seed,
    )
    informative = list(range(n_inf))  # sklearn puts informative first
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(p)])
    return df, y, informative


def _recovery(support, informative):
    s = set(support.tolist() if hasattr(support, "tolist") else support)
    inf = set(informative)
    if not inf:
        return 1.0
    return len(s & inf) / len(inf)


def _bench_method(name, fn, X, y, informative, n_runs: int = 3):
    """Run fn ``n_runs + 1`` times: discard first (warmup), report median.

    Returns (median_wall_s, recovery_fraction, support_size).
    """
    # Warmup -- first call pays the JIT-compile / pool-spawn cost.
    out = fn(X, y)

    times = []
    last_support = out
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(X, y)
        times.append(time.perf_counter() - t0)
        last_support = out

    times.sort()
    median = times[len(times) // 2]
    rec = _recovery(last_support, informative)
    print(f"  {name:42s} t={median:7.2f}s  recovery={rec * 100:5.1f}%  n_sel={len(last_support):3d}")
    return median, rec, len(last_support)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--p", type=int, default=50)
    parser.add_argument("--n-informative", type=int, default=15)
    parser.add_argument("--n-runs", type=int, default=3, help="timed runs after warmup")
    parser.add_argument("--ksg-only", action="store_true", help="skip the slower MRMR-based methods")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    print(f"\n=== n={args.n}, p={args.p}, n_inf={args.n_informative}, runs={args.n_runs} (warmup pre-pass) ===")
    print(f"  informative ground-truth indices: {list(range(args.n_informative))}")

    X, y, inf = _build(n=args.n, p=args.p, n_inf=args.n_informative, seed=args.seed)

    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.filters.estimators import (
        ksg_mi_with_target, ksg_mi_with_significance,
    )

    print("\n  estimator                                  time     recovery  n_selected")
    print("  " + "-" * 72)

    # ---- KSG paths (always run; fast on n=200k) ----
    def _ksg_topk(X, y):
        Xn = X.to_numpy() if hasattr(X, "to_numpy") else X
        mi = ksg_mi_with_target(Xn, y, feature_indices=list(range(Xn.shape[1])))
        return np.argsort(mi)[::-1][: args.n_informative]

    _bench_method("KSG top-K (no sig test)", _ksg_topk, X, y, inf, args.n_runs)

    def _ksg_sig(X, y):
        Xn = X.to_numpy() if hasattr(X, "to_numpy") else X
        _, _, sup = ksg_mi_with_significance(
            Xn, y, list(range(Xn.shape[1])),
            n_permutations=20, alpha=0.05, n_jobs=4,
        )
        return sup

    _bench_method("KSG + permutation significance test", _ksg_sig, X, y, inf, args.n_runs)

    if args.ksg_only:
        return

    # ---- MRMR paths ----
    base_kw = dict(
        quantization_nbins=10, full_npermutations=10, baseline_npermutations=5,
        n_jobs=1, verbose=0, fe_max_steps=0, cv=2,
    )

    def _mrmr_plain(X, y):
        m = MRMR(**base_kw)
        m.fit(X, y)
        return m.support_

    _bench_method("MRMR plug-in (legacy)", _mrmr_plain, X, y, inf, args.n_runs)

    # B13/B15 explicit legacy defaults to keep this diff comparable.
    def _mrmr_legacy(X, y):
        m = MRMR(max_confirmation_cand_nbins=50, fe_fallback_to_all=True, **base_kw)
        m.fit(X, y)
        return m.support_

    _bench_method("MRMR plug-in (legacy B13/B15 pinned)", _mrmr_legacy, X, y, inf, args.n_runs)

    from mlframe.feature_selection.filters.stability import StabilityMRMR

    def _stability(X, y):
        s = StabilityMRMR(estimator=MRMR(**base_kw), n_bootstraps=5, sample_fraction=0.7, support_threshold=0.6, random_state=42)
        s.fit(X, y)
        return s.support_

    _bench_method("StabilityMRMR (B=5)", _stability, X, y, inf, args.n_runs)

    from mlframe.feature_selection.filters.group_aware import GroupAwareMRMR

    def _group(X, y):
        g = GroupAwareMRMR(estimator=MRMR(**base_kw), corr_threshold=0.9)
        g.fit(X, y)
        return g.support_

    _bench_method("GroupAwareMRMR (corr>=0.9)", _group, X, y, inf, args.n_runs)


if __name__ == "__main__":
    main()
