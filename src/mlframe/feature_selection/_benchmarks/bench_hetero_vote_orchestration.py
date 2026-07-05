"""cProfile-driven orchestration profile for ``heterogeneous_relevance_vote`` (hetero_vote.py).

Run: ``PYTHONPATH=src CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python -m mlframe.feature_selection._benchmarks.bench_hetero_vote_orchestration``

WHAT THIS PROFILES: the SELECTOR-ORCHESTRATION around the panel/shadow loop -- NOT the sklearn estimator
fits (RandomForest/Ridge/KNN .fit) nor sklearn permutation_importance, which are sklearn-bound and out of
scope. The orchestration seams under audit (hetero_vote.py):
  - line 122  shadow build: ``np.column_stack([rng.permutation(Xv[:, j]) for j in range(P)])`` -- a Python
              loop over P columns, run once PER (model x trial) = n_models * n_shadow_trials times.
  - line 123  ``np.hstack([Xv, shadow])`` -- rebuilds the full [X|shadow] matrix every (model x trial).
  - line 133  vote aggregation ``(W[:, None] * np.vstack(passes)).sum(axis=0)`` -- a vstack across models.
  - line 39   ``np.asarray(y)[idx]`` inside _importance (perm-importance fallback only; the default panel
              has feature_importances_/coef_ so this rarely fires).

KEY OBSERVATION (line 122): the shadow for model m, trial t is ``column_stack([rng.permutation(Xv[:,j]) ...])``
with ``rng = default_rng(random_state + tr)``. The seed depends ONLY on ``tr``, NOT on the model. So model A
trial t and model B trial t draw the *identical* shadow matrix from scratch. With n_models=3 and n_trials=3
the same n_trials shadow matrices are rebuilt n_models times each -- (n_models-1)/n_models = 2/3 of the shadow
construction work is redundant recomputation of a loop-invariant. Hoisting shadow build out of the panel loop
(precompute the n_trials shadows once, reuse across all models) removes that redundancy BIT-IDENTICALLY (same
seed -> same permutation -> same array), with the win scaling as n_models.

REPRESENTATIVE SHAPE (per task brief): regression panel, n~2000, p~20, 3-model panel, modest n_perm_repeats.
A small shape (n=500, p=10) is also profiled to confirm the orchestration share holds at small n.

MEASURED VERDICT (2026-06-11): RESOLVED -- shadow-build hoist shipped, bit-identical.
  Profile (n=2000, p=20, 3 trials): total wall ~32s, of which ~27.4s (86%) is the KNN distance reduction
  inside sklearn permutation_importance (the default panel's distance member has neither feature_importances_
  nor coef_, so it takes the perm-importance fallback). ``heterogeneous_relevance_vote`` orchestration tottime
  is 0.033s; the shadow build / hstack / vote-aggregation calls do not appear in the top-28 by tottime (they
  are <0.1% of wall here, dominated by the sklearn KNN kernel which is OUT OF SCOPE).
  Orchestration audit findings (tottime / callcount):
    1. shadow construction (line 122 + hstack 123) -- redundant LOOP-INVARIANT recompute: n_models * n_trials
       builds where n_trials distinct shadows suffice (seed depends only on tr). 2/3 of the work was redundant.
       MICROBENCH (isolated, warm, 30 reps, n_models=3): 2.43x @ n2000/p20, 2.26x @ n500/p10, 2.87x @ n5000/p40.
       FIXED: hoisted the n_trials [X|shadow] builds out of the panel loop and reuse across members.
       BIT-IDENTICAL (vote_fraction byte-equal + accepted set equal) verified pre-vs-post on reg+clf x 2 seeds
       x weight_by_cv_skill in {False,True} = 8 cells, all IDENTICAL.
    2. vote aggregation (line 133, np.vstack(passes) then weighted sum) -- runs ONCE per call (not per model);
       <1ms standalone, attribution noise. NOT optimized (already a single vectorized reduction).
    3. _importance np.asarray(y)[idx] (line 39) -- only fires on the perm-importance fallback member; yv is
       already an ndarray so np.asarray is a no-op view; the [idx] gather is required (subsample). NOT a hotspot.
  End-to-end the hoist is a tiny fraction of wall on the DEFAULT panel (KNN-bound), but it is genuine redundant
  work removed, bit-identical by construction, the code is cleaner (invariant computed once), and the win scales
  with n_models AND grows when the panel members are cheap (all tree/linear, no KNN) -- exactly the case where
  orchestration dominates. Shipped per the small-clean-win + loop-invariant-hoist standards.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.hetero_vote import heterogeneous_relevance_vote


def make_reg(seed: int, n: int = 2000, p_sig: int = 5, p_noise: int = 15) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, p_sig))
    beta = np.array([1.5, -1.2, 1.0, 0.9, -0.7])[:p_sig]
    y = z @ beta + 0.5 * rng.standard_normal(n)
    cols = {f"sig_{i}": z[:, i] for i in range(p_sig)}
    for j in range(p_noise):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y)


def _run(X, y, n_trials=3):
    return heterogeneous_relevance_vote(X, y, classification=False, n_shadow_trials=n_trials, random_state=0)


def profile_shape(name: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 3) -> None:
    pr = cProfile.Profile()
    pr.enable()
    _run(X, y, n_trials)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    print(f"\n{'='*90}\n{name}: n={X.shape[0]} p={X.shape[1]} trials={n_trials}\n{'='*90}")
    print("--- by CUMULATIVE ---")
    ps.sort_stats("cumulative").print_stats(28)
    print("--- by TOTTIME ---")
    ps.sort_stats("tottime").print_stats(28)
    print(s.getvalue())


def microbench() -> None:
    """Isolated warm wall-time of the shadow-build loop-invariant vs the current per-model rebuild."""
    print(f"\n{'='*90}\nMICROBENCH: shadow construction (line 122), n_models=3\n{'='*90}")
    for n, p in [(2000, 20), (500, 10), (5000, 40)]:
        rng_data = np.random.default_rng(0)
        Xv = rng_data.standard_normal((n, p))
        n_models, n_trials, reps = 3, 3, 30

        def current():
            for _m in range(n_models):
                for tr in range(n_trials):
                    rng = np.random.default_rng(tr)
                    np.column_stack([rng.permutation(Xv[:, j]) for j in range(p)])

        def hoisted():
            shadows = [np.column_stack([np.random.default_rng(tr).permutation(Xv[:, j]) for j in range(p)]) for tr in range(n_trials)]
            for _m in range(n_models):
                for tr in range(n_trials):
                    _ = shadows[tr]

        current(); hoisted()  # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            current()
        t_cur = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            hoisted()
        t_hoist = (time.perf_counter() - t0) / reps
        print(f"  n={n:5d} p={p:3d}: current={t_cur*1e3:7.3f}ms  hoisted={t_hoist*1e3:7.3f}ms  " f"speedup={t_cur/t_hoist:5.2f}x")


if __name__ == "__main__":
    Xr, yr = make_reg(0)
    profile_shape("REPRESENTATIVE regression", Xr, yr, n_trials=3)
    Xs, ys = make_reg(0, n=500, p_sig=3, p_noise=7)
    profile_shape("SMALL regression", Xs, ys, n_trials=3)
    microbench()
