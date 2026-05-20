"""Clean polynom-FE bench (no tqdm clutter) - diagnoses the "0 features"
mystery from the previous backend bench.

Writes to stdout with NO progress bars; redirect to a file for clean
single-shot results.
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Silence tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"


def _build_problem(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.normal(size=n).astype(np.float64) for i in range(6)}
    # Strong multiplicative interaction so polynom-FE SHOULD find it.
    z = (cols["x0"] + cols["x1"] + 2.0 * cols["x0"] * cols["x1"]
         + 0.5 * cols["x2"] * cols["x3"] + rng.normal(0, 0.3, n))
    y = (z > np.median(z)).astype(np.int64)
    return pd.DataFrame(cols), y


def _run_mrmr(*, X, y, n_jobs, subsample_n, label) -> dict:
    from mlframe.feature_selection.filters.mrmr import MRMR

    # Hygiene for valid cross-case comparison (see bench_polynom_fe_backends.py
    # commit msg 2026-05-18 for the original "njit_par 135s" anomaly diagnosis):
    # 1. clear fit cache so prior runs don't HIT
    # 2. random_seed=42 pins CMA-ES trajectory
    # 3. X.copy() prevents polynom-FE column-injection from leaking to next case
    MRMR.clear_fit_cache()
    t0 = time.perf_counter()
    m = MRMR(
        fe_smart_polynom_iters=2,
        fe_smart_polynom_optimization_steps=30,
        fe_min_polynom_degree=1, fe_max_polynom_degree=3,
        fe_max_pair_features=4, fe_min_pair_mi=-1.0,
        fe_min_pair_mi_prevalence=0.0,
        fe_min_engineered_mi_prevalence=0.0,
        fe_min_nonzero_confidence=0.0, min_nonzero_confidence=0.0,
        fe_smart_polynom_subsample_n=subsample_n,
        random_seed=42,
        n_jobs=n_jobs, verbose=0,
        mrmr_skip_when_prior_was_identity=False,
    )
    m.fit(X.copy(), y)
    t = time.perf_counter() - t0
    return {
        "label": label,
        "time_s": t,
        "n_hermite": len(getattr(m, "_hermite_features_", None) or []),
        "n_engineered_recipes": len(getattr(m, "_engineered_recipes_", []) or []),
        "n_support": len(m.support_),
        "support_names": [m.feature_names_in_[i] for i in m.support_],
    }


def main(n_rows: int = 1_000_000) -> int:
    logging.getLogger("mlframe").setLevel(logging.WARNING)
    print(f"\n{'='*70}\nPolynom-FE clean bench at n={n_rows:_}\n{'='*70}")
    print(f"CPU count: {os.cpu_count()}")
    try:
        import cupy as _cp  # noqa: F401
        print("cupy / CUDA: available")
    except Exception as e:
        print(f"cupy / CUDA: NOT available ({type(e).__name__}: {str(e)[:80]})")
    print()

    X, y = _build_problem(n=n_rows)
    n_feats = X.shape[1]
    print(f"Problem: n={len(X):_}, features={n_feats} "
          f"({n_feats*(n_feats-1)//2} candidate pairs), "
          f"target_balance={np.bincount(y).tolist()}")
    print()

    # Warmup
    X_w, y_w = _build_problem(n=2000, seed=1)
    _run_mrmr(X=X_w, y=y_w, n_jobs=1, subsample_n=0, label="warmup")
    print("[JIT warmup done]")
    print()

    # subsample=200_000 matches the post-2026-05-18 MRMR default
    # (``fe_smart_polynom_subsample_n``). At 100k a non-trivial fraction
    # of CMA-ES trials lose the genuine hermite=1 feature -- raising the
    # subsample to 200k recovered it. The 100k row stays for historical
    # comparison; new runs should focus on the 200k case.
    cases = [
        ("subsample=0     n_jobs=1",                  0,        1, None),
        ("subsample=100k  n_jobs=4",                  100_000,  4, None),
        ("subsample=200k  n_jobs=1",                  200_000,  1, None),
        ("subsample=200k  n_jobs=4 (production)",     200_000,  4, None),
        ("subsample=200k  n_jobs=4 backend=njit",     200_000,  4, "njit"),
        ("subsample=200k  n_jobs=4 backend=njit_par", 200_000,  4, "njit_par"),
    ]
    if "cupy" in sys.modules or True:  # add cuda case always, will fall back gracefully
        cases.append(("subsample=200k  n_jobs=4 backend=cuda", 200_000, 4, "cuda"))

    print(f"{'Variant':45s} | {'Time':>8s} | {'Hermite':>7s} | {'Recipes':>7s} | {'Support':>7s}")
    print("-" * 100)
    for label, ssn, nj, backend in cases:
        if backend:
            os.environ["MLFRAME_POLYEVAL_BACKEND"] = backend
        else:
            os.environ.pop("MLFRAME_POLYEVAL_BACKEND", None)
        try:
            r = _run_mrmr(X=X, y=y, n_jobs=nj, subsample_n=ssn, label=label)
            print(f"{r['label']:45s} | {r['time_s']:7.2f}s | "
                  f"{r['n_hermite']:>7d} | {r['n_engineered_recipes']:>7d} | "
                  f"{r['n_support']:>7d}")
        except Exception as e:
            print(f"{label:45s} | FAILED: {type(e).__name__}: {str(e)[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
