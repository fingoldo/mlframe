"""Full-fit A/B: FE-decision f64 upcast vs f32 relaxation (MLFRAME_CRIT_DTYPE_RELAXED) on the batched-MI ranking.

The orth-FE families materialised their seed-ranking matrix with a hardcoded ``X[cols].to_numpy(np.float64)``. On f32
source data (the common case -- e.g. the wellbore frame) that is a pure upcast COPY + wider MI compute. This routes the
dominant batched-MI sites through the existing ``_crit_np_dtype()`` knob (f32 by default). This bench measures the REALISED
full-``MRMR.fit`` wall of strict-f64 (MLFRAME_CRIT_DTYPE_RELAXED=0) vs relaxed-f32 (=1) on the SAME f32 frame, alternating
for paired timing, and asserts the SELECTION is unchanged (the FE/MRMR bar is selection-equivalence, not bit-identity).

Run: python -m mlframe.feature_selection._benchmarks.bench_fe_crit_dtype_relax_fullfit
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

N = 80_000
K = 250
TRIALS = 3


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, K)).astype(np.float32)  # f32 source -> f64 path is a pure upcast copy
    y = (X[:, 0] * 1.3 + X[:, 5] * X[:, 9] + rng.standard_normal(N) * 0.4 > 0).astype(np.int64)
    cols = [f"f{i}" for i in range(K)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def _fit_once(X, y):
    from mlframe.feature_selection.filters.mrmr import MRMR
    MRMR._FIT_CACHE.clear()  # force a real re-fit (MRMR memoises by content hash)
    m = MRMR(
        fe_univariate_basis_enable=True,
        fe_hybrid_orth_enable=False,
        fe_max_steps=1,
        verbose=0,
        random_seed=0,
    )
    t = time.perf_counter()
    m.fit(X, y)
    dt = time.perf_counter() - t
    return dt, sorted(map(str, m.support_))


def _run(flag, X, y):
    os.environ["MLFRAME_CRIT_DTYPE_RELAXED"] = flag  # "0" -> strict f64, "1" -> relaxed f32
    return _fit_once(X, y)


def main():
    X, y = _data()
    # Warm numba / cache once per mode (first fit compiles; not timed).
    _run("0", X, y)
    _run("1", X, y)

    f64_times, f32_times = [], []
    sel64 = sel32 = None
    for _ in range(TRIALS):
        dt64, sel64 = _run("0", X, y)   # alternate f64/f32 so shared-machine load cancels in the paired compare
        dt32, sel32 = _run("1", X, y)
        f64_times.append(dt64)
        f32_times.append(dt32)

    m64, m32 = float(np.median(f64_times)), float(np.median(f32_times))
    print(f"\nFE crit-dtype full-fit A/B  n={N} k={K} trials={TRIALS}")
    print(f"  f64 (strict)  median {m64:.2f}s   times={[round(t, 2) for t in f64_times]}")
    print(f"  f32 (relaxed) median {m32:.2f}s   times={[round(t, 2) for t in f32_times]}")
    print(f"  speedup (f64/f32) = {m64 / m32:.3f}x")
    print(f"  selection identical (f32 == f64): {sel32 == sel64}  (|sel64|={len(sel64)} |sel32|={len(sel32)})")
    if sel32 != sel64:
        only64 = set(sel64) - set(sel32)
        only32 = set(sel32) - set(sel64)
        print(f"    only-f64={sorted(only64)[:8]}  only-f32={sorted(only32)[:8]}")


if __name__ == "__main__":
    main()
