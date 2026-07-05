"""A/B bench for the ``_fe_usability_signal._abs_pearson_njit`` fastmath+branchless optimisation (2026-07).

RESOLVED: branchless select + ``fastmath={'reassoc','contract','arcp','afn','nsz'}`` (nnan/ninf DELIBERATELY
excluded so the NaN-drop survives) is 2.1-2.5x over the old plain-branch ``fastmath=False`` kernel at every
n in 600..30000, selection-equivalent (diff <=~1e-16 single ULP, NaN rows dropped EXACTLY).

Also documents two REJECTED alternatives so the negative results are reproducible:
  * batch-9-forms-into-1-njit-dispatch: 0.33x at n=30000 -- compute-bound not dispatch-bound; the f64 matrix
    materialisation + strided column writes cost more than the 9 saved dispatches.
  * full ``fastmath=True``: 4x BUT silently returns 0.0 on NaN data (LLVM drops the isfinite test) -> a
    selection-breaking ~1e-2 error. UNSAFE.

Run:  python path/to/_benchmarks/bench_abs_pearson_fastmath.py
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from mlframe.feature_selection.filters._fe_usability_signal import _abs_pearson_njit, abs_pearson  # noqa: F401


# The pre-fix baseline (plain branch, fastmath=False) reconstructed for the A/B OLD side.
@njit(cache=True, fastmath=False)
def _abs_pearson_baseline(y, v):
    n = 0
    sa = 0.0; sv = 0.0; saa = 0.0; svv = 0.0; sav = 0.0
    for i in range(y.shape[0]):
        a = np.float64(y[i]); b = np.float64(v[i])
        if np.isfinite(a) and np.isfinite(b):
            n += 1
            sa += a; sv += b; saa += a * a; svv += b * b; sav += a * b
    if n < 2:
        return 0.0
    inv = 1.0 / n
    va = saa - sa * sa * inv
    vv2 = svv - sv * sv * inv
    if va <= 0.0 or vv2 <= 0.0:
        return 0.0
    den = (va * vv2) ** 0.5
    if den <= 0.0:
        return 0.0
    c = (sav - sa * sv * inv) / den
    if not np.isfinite(c):
        return 0.0
    return -c if c < 0.0 else c


def main() -> None:
    rng = np.random.default_rng(1)
    R = 3000
    print(f"{'n':>7} {'nan':>5} {'old_us':>9} {'new_us':>9} {'speedup':>8} {'maxdiff':>10}")
    for n in (600, 5000, 30000):
        for frac in (0.0, 0.1):
            y = rng.standard_normal(n).astype(np.float32)
            v = rng.standard_normal(n).astype(np.float32).copy()
            if frac > 0:
                v[rng.choice(n, int(n * frac), replace=False)] = np.nan
            old = _abs_pearson_baseline(y, v)
            new = _abs_pearson_njit(y, v)
            diff = abs(old - new)
            t = time.perf_counter()
            for _ in range(R):
                _abs_pearson_baseline(y, v)
            t_old = (time.perf_counter() - t) / R * 1e6
            t = time.perf_counter()
            for _ in range(R):
                _abs_pearson_njit(y, v)
            t_new = (time.perf_counter() - t) / R * 1e6
            print(f"{n:>7} {frac:>5.0%} {t_old:>9.1f} {t_new:>9.1f} {t_old / t_new:>7.2f}x {diff:>10.1e}")


if __name__ == "__main__":
    main()
