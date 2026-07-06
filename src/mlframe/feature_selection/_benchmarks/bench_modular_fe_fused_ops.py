"""Bench: fused 3-op modular kernel vs three separate apply_modular passes in generate_modular_features.

OLD path: for each (col, period) calls apply_modular() three times (mod / sin / cos). Each call re-runs
np.ascontiguousarray + a full njit pass that recomputes the residue r = v - p*floor(v/p) independently.

NEW path: _modular_all_ops_njit() computes the residue ONCE per element and emits all three outputs in a
single pass per (col, period). Bit-identical by construction (same residue formula, same NaN/inf scrub,
same trig of the same phase).

Run: python -m mlframe.feature_selection._benchmarks.bench_modular_fe_fused_ops
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._periodic_fe import (
    _modular_njit,
    _modular_all_ops_njit,
    _VALID_OPS,
    engineered_name_modular,
    DEFAULT_PERIODS,
    generate_modular_features,
)


def _old_generate(X, periods):
    out = {}
    for c in X.columns:
        x = np.ascontiguousarray(X[c].to_numpy(), dtype=np.float64)
        for p in periods:
            for op in _VALID_OPS:
                oc = 0 if op == "mod" else (1 if op == "sin" else 2)
                out[engineered_name_modular(c, p, op)] = _modular_njit(x, float(p), oc)
    return pd.DataFrame(out, index=X.index)


def _best_of(fn, n=7):
    fn()  # warm
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    for n, ncols in ((50_000, 8), (200_000, 12)):
        X = pd.DataFrame({f"f{j}": rng.normal(0, 100, n) for j in range(ncols)})
        # sprinkle non-finite
        X.iloc[::997, 0] = np.nan
        X.iloc[::1013, 1] = np.inf
        periods = DEFAULT_PERIODS

        old = _best_of(lambda: _old_generate(X, periods))
        new = _best_of(lambda: generate_modular_features(X, periods=periods))

        # identity
        a = _old_generate(X, periods)
        b = generate_modular_features(X, periods=periods)
        assert list(a.columns) == list(b.columns)  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input
        max_abs = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))
        exact = bool(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))

        print(f"n={n:>7} cols={ncols:>2}  OLD={old*1e3:8.2f}ms  NEW={new*1e3:8.2f}ms  " f"speedup={old/new:5.2f}x  max|diff|={max_abs:.2e}  exact_eq={exact}")


if __name__ == "__main__":
    main()
