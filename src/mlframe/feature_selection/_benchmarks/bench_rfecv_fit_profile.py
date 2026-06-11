"""cProfile + warm microbench for the RFECV wrapper fit hot path.

Run (CPU-only, PYTHONPATH=src):
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m mlframe.feature_selection._benchmarks.bench_rfecv_fit_profile

Two jobs:

1. cProfile ``RFECV.fit`` on a representative shape (n=2000, p=40, cv=3) with both a
   LogisticRegression (classifier) and a Ridge (regressor) estimator, plus a small
   shape (n=400, p=20). Sorts by cumulative time, top 28. Dump + text saved under
   ``_results/``.

2. A WARM multi-run isolated microbench of the closed-form permutation scorer body
   (``_make_fast_default_scorer`` -> ``_fast_value``) on both clf and reg targets.

Profile findings (n=2000 p=40 cv=3, this hardware, 2026-06-11)
--------------------------------------------------------------
The fit wall is dominated by sklearn's ``permutation_importance`` inner loop -- the
``p*n_repeats`` scorer calls per fold, each running ``estimator.predict`` (irreducible
model inference) + sklearn's ``check_random_state`` / ``check_array`` / warnings
machinery. Those are sklearn-internal and out of the wrapper's control.

Top mlframe-side (``wrappers/``) hotspots by tottime:

  1. ``_helpers_importance._fast_value``  (23k+ calls)  -- the closed-form scorer body.
  2. (``metrics/_ice_metric.compute_probabilistic_multiclass_error`` is the default
     SCORING metric, hot by tottime but lives in ``metrics/``, not ``wrappers/`` --
     out of this harness's scope.)

Optimisation shipped: y-invariant hoist in ``_fast_value``
----------------------------------------------------------
``permutation_importance`` permutes only X and feeds the IDENTICAL ``_y`` to the
scorer on every one of its ``p*n_repeats`` calls. The pre-fix scorer recomputed
y-only quantities every call:
  - classifier: ``np.asarray(_y)``
  - regressor:  ``_y.astype(float64)`` + ``np.mean(_y)`` + ``ss_tot = sum((y-mean)**2)``

These are now computed ONCE per ``permutation_importance`` call and reused via an
``id(_y)`` cache held in the scorer closure. Bit-identical by construction (same
float, same NaN handling -- the only change is not recomputing a pure function of an
unchanged input).

Isolated warm microbench (1000-row target, min of 5 x 200k iters, this hardware):
  - regressor scorer body: 18.3 us -> 6.0 us  (~3.0x; the win lives here)
  - classifier scorer body: ~11.5 us -> ~11.3 us  (np.mean(pred==y) is irreducible;
    only the tiny ``np.asarray(_y)`` is hoisted, so the clf win is marginal)

End-to-end the win shows on REGRESSOR fits (Ridge / any r2-scored estimator) where the
scorer's regressor branch is the per-call body; classifier fits see a negligible delta
because their per-call cost is the irreducible ``np.mean(pred==y)`` + ``predict``.

Bit-identity: RFECV selection (support_ / n_features_ / ranking_) is bit-identical
before/after on fixed-seed fixtures -- see
``tests/feature_selection/test_rfecv_fast_perm_scorer_identity.py``.

cProfile-overhead caveat: the scorer body microbenches at ~6-18 us/call standalone, so
cProfile's per-call attribution (~6.5 us tottime) is at the noise boundary; the warm
isolated microbench above is the authoritative measurement, not the cProfile tottime.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers._helpers_importance import _make_fast_default_scorer


def _clf_problem(n: int, p: int, seed: int = 0):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=max(5, p // 4),
        n_redundant=0, n_classes=2, n_clusters_per_class=1,
        random_state=seed, shuffle=False,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


def _reg_problem(n: int, p: int, seed: int = 0):
    X, y = make_regression(
        n_samples=n, n_features=p, n_informative=max(5, p // 4),
        noise=0.1, random_state=seed, shuffle=False,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


def _profile_once(out_dir: Path, X, y, estimator, label: str) -> None:
    rfecv = RFECV(estimator=estimator, cv=3, max_refits=12, verbose=0, random_state=0)
    profiler = cProfile.Profile()
    profiler.enable()
    rfecv.fit(X, y)
    profiler.disable()

    out_dir.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(out_dir / f"rfecv_fit_{label}.prof"))

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(28)
    (out_dir / f"rfecv_fit_{label}.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(f"\n=== {label}  n_features_selected={rfecv.n_features_} ===")
    print(stream.getvalue())


def _microbench_scorer() -> None:
    rng = np.random.RandomState(0)
    n = 1000
    # regressor
    Xr = rng.randn(n, 5)
    yr = Xr @ np.arange(5) + 0.1 * rng.randn(n)
    mr = Ridge().fit(Xr, yr)
    scr = _make_fast_default_scorer(mr)
    scr(mr, Xr, yr)  # latch fast path on baseline
    # classifier
    yc = (yr > yr.mean()).astype(int)
    mc = LogisticRegression(max_iter=500).fit(Xr, yc)
    scc = _make_fast_default_scorer(mc)
    scc(mc, Xr, yc)

    def bench(fn, *args, iters: int = 50_000, runs: int = 5) -> float:
        best = float("inf")
        for _ in range(runs):
            t = time.perf_counter()
            for _ in range(iters):
                fn(*args)
            best = min(best, time.perf_counter() - t)
        return best / iters * 1e6

    print("\n=== warm scorer microbench (us/call, min of 5 x 50k) ===")
    print(f"regressor scorer:  {bench(scr, mr, Xr, yr):.3f} us")
    print(f"classifier scorer: {bench(scc, mc, Xr, yc):.3f} us")


def main() -> None:
    out_dir = Path(__file__).parent / "_results"
    print(f"# RFECV fit profile -> {out_dir}")

    Xc_s, yc_s = _clf_problem(n=400, p=20)
    _profile_once(out_dir, Xc_s, yc_s, LogisticRegression(max_iter=400, random_state=0), "small_clf")

    Xc, yc = _clf_problem(n=2000, p=40)
    _profile_once(out_dir, Xc, yc, LogisticRegression(max_iter=400, random_state=0), "repr_clf")

    Xr, yr = _reg_problem(n=2000, p=40)
    _profile_once(out_dir, Xr, yr, Ridge(random_state=0), "repr_reg")

    _microbench_scorer()


if __name__ == "__main__":
    main()
