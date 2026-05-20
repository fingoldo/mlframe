"""Backend / subsample bench for polynom-FE Hermite kernels at n=1M.

Investigates the user's question: "can we speed up further with
numba.parallel / numba.cuda / cupy?".

Existing infrastructure (`polyeval_dispatch` in hermite_fe.py):
- n < 50k: ``_hermeval_njit`` single-thread Horner
- 50k <= n < 500k: ``_hermeval_njit_parallel`` (prange) - 1.5-2x
- n >= 500k: ``_polyeval_cuda`` (cupy RawKernel) - ~5x over njit_par
  (if cupy available; silent fallback otherwise)

The MI estimator ``_plugin_mi_classif_njit`` is single-thread numba;
``_plugin_mi_classif_batch_njit`` parallels over the (typically 3) binary
function columns. No CUDA path for MI.

Two axes:
- ``fe_smart_polynom_subsample_n`` ON/OFF (default 100k since 2026-05-18)
- env ``MLFRAME_POLYEVAL_BACKEND`` override (cuda / njit_par / njit)

Bench: n=1M source, fe_smart_polynom_iters=2 x 30 trials per pair, 4
pairs total via fe_max_pair_features.

Run: python profiling/bench_polynom_fe_backends.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd


def _build_problem(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.normal(size=n).astype(np.float64) for i in range(6)}
    z = (cols["x0"] + cols["x1"] + 2.0 * cols["x0"] * cols["x1"]
         + 0.5 * cols["x2"] * cols["x3"] + rng.normal(0, 0.3, n))
    y = (z > np.median(z)).astype(np.int64)
    return pd.DataFrame(cols), y


def _run_mrmr(*, X, y, n_jobs, subsample_n) -> tuple[float, int]:
    from mlframe.feature_selection.filters.mrmr import MRMR

    # CRITICAL: clear class-level fit-cache so prior runs don't HIT.
    # CRITICAL: ``random_seed=42`` so CMA-ES trajectory is deterministic
    # across cases - without it, MRMR derives seed from pid ^ id(self)
    # which varies per call and produces variance-of-30x between runs
    # (verified 2026-05-18 - the original 135s "njit_par anomaly" was
    # entirely explained by random_seed variation, NOT polyeval bug).
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
    # CRITICAL: fresh X clone per call - the polynom-FE block mutates X
    # by adding ``_polynom_*`` columns (polynom_pair_fe.py:206
    # ``X[_new_col_name] = _t_vals``). Shared X across cases produces
    # accumulating col-count and DIFFERENT prospective pairs - the
    # original bench's "cuda OOM at 7.63 MiB" was this leak compounded
    # with cupy memory pool fragmentation across cases.
    m.fit(X.copy(), y)
    return time.perf_counter() - t0, len(getattr(m, "_hermite_features_", None) or [])


def main() -> int:
    print("=" * 70)
    print("Polynom-FE backends + subsample bench at n=1M")
    print("=" * 70)
    print(f"CPU count: {os.cpu_count()}")
    cuda_avail = False
    try:
        import cupy as _cp  # noqa: F401
        cuda_avail = True
    except Exception:
        pass
    print(f"cupy / CUDA: {'available' if cuda_avail else 'NOT available'}")
    print()

    X, y = _build_problem(n=1_000_000)
    print(f"Problem: n={len(X):_}, n_feats={X.shape[1]} "
          f"(~{X.shape[1]*(X.shape[1]-1)//2} pairs)")
    print()

    # Warm up JIT once on a tiny frame so dispatcher caches kernels.
    X_warm, y_warm = _build_problem(n=2000, seed=1)
    _run_mrmr(X=X_warm, y=y_warm, n_jobs=1, subsample_n=0)
    print("[JIT warmup done]")
    print()

    matrix = [
        ("no_subsample, n_jobs=1", 0, 1, None),
        ("subsample=100k, n_jobs=1", 100_000, 1, None),
        ("subsample=100k, n_jobs=4 (default)", 100_000, 4, None),
        ("no_subsample, n_jobs=4, backend=njit", 0, 4, "njit"),
        ("no_subsample, n_jobs=4, backend=njit_par", 0, 4, "njit_par"),
    ]
    if cuda_avail:
        matrix.append(
            ("no_subsample, n_jobs=4, backend=cuda", 0, 4, "cuda")
        )

    for label, ssn, nj, backend in matrix:
        if backend is not None:
            os.environ["MLFRAME_POLYEVAL_BACKEND"] = backend
        else:
            os.environ.pop("MLFRAME_POLYEVAL_BACKEND", None)
        try:
            t, nh = _run_mrmr(X=X, y=y, n_jobs=nj, subsample_n=ssn)
            print(f"  {label:50s} -> {t:6.2f}s  hermite_features={nh}")
        except Exception as e:
            print(f"  {label:50s} -> FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
