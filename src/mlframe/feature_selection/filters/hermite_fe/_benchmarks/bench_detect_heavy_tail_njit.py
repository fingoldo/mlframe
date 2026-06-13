"""Isolation microbench: _detect_heavy_tail numpy vs njit.

The predicate is called per-column many times inside the FE preprocess/warp search loop (10098 calls / 11.0s cumtime in
the MRMR scene profile, iter49). The numpy body walks the finite subset several times: isfinite-filter, median (sort),
MAD median (second sort), abs-dev recompute, threshold count, masked max + masked min. An njit single-pass kernel fuses
the dev pass + masked reductions and keeps the two medians (unavoidable sorts) inside one compiled frame, removing the
Python-level temporaries (boolean masks, intermediate float64 arrays) the numpy path allocates each call.

Run: PYTHONPATH=<worktree>/src python bench_detect_heavy_tail_njit.py
"""
from __future__ import annotations
import time
import numpy as np

from mlframe.feature_selection.filters.hermite_fe._hermite_robust import (
    _detect_heavy_tail,
    _detect_heavy_tail_njit,
    _ROBUST_AXIS_OUTER_K,
    _ROBUST_AXIS_GAP,
    _ROBUST_AXIS_MAX_FRAC,
)


def _make_cols(rng, n):
    cols = []
    # clean gaussian (no trip)
    cols.append(rng.standard_normal(n))
    # lognormal heavy tail (no trip - continuous)
    cols.append(rng.lognormal(0.0, 1.0, n))
    # spike contamination (trip)
    x = rng.standard_normal(n)
    idx = rng.choice(n, size=max(1, n // 200), replace=False)
    x[idx] += 50.0
    cols.append(x)
    # with NaNs
    x = rng.standard_normal(n)
    x[rng.choice(n, size=n // 10, replace=False)] = np.nan
    cols.append(x)
    # near-constant
    cols.append(np.full(n, 3.0) + rng.standard_normal(n) * 1e-15)
    return [c.astype(np.float64) for c in cols]


def main():
    rng = np.random.default_rng(0)
    for n in (1000, 2407, 4000, 6000, 8000, 10000, 50000):
        cols = _make_cols(rng, n)
        # bit-identity check
        for c in cols:
            a = _detect_heavy_tail(c)
            b = _detect_heavy_tail_njit(c)
            assert a == b, f"MISMATCH n={n}: numpy={a} njit={b}"
        # warm njit
        for c in cols:
            _detect_heavy_tail_njit(c)

        reps = 2000
        t0 = time.perf_counter()
        for _ in range(reps):
            for c in cols:
                _detect_heavy_tail(c)
        t_np = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(reps):
            for c in cols:
                _detect_heavy_tail_njit(c)
        t_nj = time.perf_counter() - t0

        per_np = t_np / (reps * len(cols)) * 1e6
        per_nj = t_nj / (reps * len(cols)) * 1e6
        print(f"n={n:6d}  numpy={per_np:8.2f}us  njit={per_nj:8.2f}us  speedup={per_np / per_nj:5.2f}x  (bit-identical)")


if __name__ == "__main__":
    main()
