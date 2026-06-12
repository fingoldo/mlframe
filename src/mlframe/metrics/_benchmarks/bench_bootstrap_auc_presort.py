"""Bench: pre-sort base score vector ONCE + O(n) counting-gather per bootstrap
resample, vs a full np.argsort of the resampled vector every iteration.

fast_roc_auc_unstable is called per bootstrap resample (1000x) and re-argsorts
the n-length resampled score vector each time -> 1000x O(n log n). Pre-argsort
the base ONCE, then build each resample's descending order via an O(n) counting
sort over the precomputed base rank. Bit-identical on tie-free float64 scores
(GATE on all-distinct); tied/discrete base routes to the exact argsort path.

Run: python -m mlframe.metrics._benchmarks.bench_bootstrap_auc_presort
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.metrics._core_auc_brier import (
    fast_roc_auc_unstable,
    make_bootstrap_auc_resampler,
    fast_numba_auc_nonw,
)


def _bench(n: int, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    y_score = rng.random(n)  # continuous float64: all-distinct (tie-free)
    y_true = (rng.random(n) < 0.3).astype(np.int64)

    # Precompute the resample index sets ONCE so both methods see identical idx.
    idxs = [rng.integers(0, n, size=n, dtype=np.int64) for _ in range(n_boot)]

    # --- OLD: argsort the resampled vector each iter ---
    t0 = time.perf_counter()
    old_vals = np.empty(n_boot)
    for k, idx in enumerate(idxs):
        old_vals[k] = fast_roc_auc_unstable(y_true[idx], y_score[idx])
    old_ms = (time.perf_counter() - t0) * 1e3

    # --- NEW: pre-sort base once, counting-gather per resample ---
    resampler = make_bootstrap_auc_resampler(y_true, y_score)
    t1 = time.perf_counter()
    new_vals = np.empty(n_boot)
    for k, idx in enumerate(idxs):
        new_vals[k] = resampler(idx)
    new_ms = (time.perf_counter() - t1) * 1e3

    max_abs = float(np.max(np.abs(old_vals - new_vals)))
    return {"n": n, "old_ms": old_ms, "new_ms": new_ms,
            "speedup": old_ms / new_ms, "max_abs_diff": max_abs}


if __name__ == "__main__":
    # warm JIT
    _bench(2000, n_boot=5)
    for n in (5000, 20000, 50000, 200000):
        r = _bench(n)
        print(f"n={r['n']:>7}  old={r['old_ms']:8.1f}ms  new={r['new_ms']:8.1f}ms  "
              f"speedup={r['speedup']:.2f}x  max|diff|={r['max_abs_diff']:.2e}")
