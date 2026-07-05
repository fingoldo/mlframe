"""Bench (lead5): float64 pre-cast outside the bootstrap resample loop.

``honest_diagnostics._bootstrap_block`` ``_brier`` / ``_ll`` used to cast yy/pp
to float64 INSIDE the 1000-resample loop; on int labels that copies every call
(~4000 in-loop copies / block). The fix pre-casts y_true / p_pos to float64 ONCE
before the loop so the resampled views are already float64 and the kernels skip
the per-call cast. Bit-identical (same numbers into the same numba kernels).

Run: python -m mlframe.training._benchmarks.bench_bootstrap_block_precast

Measured (n=20000, stratified, 1000 bootstrap, 4 metrics, warm, 12 reps, dev box):
  old (int arrays + in-loop astype in brier/ll): ~1338.7 ms / block
  new (pre-cast float64, bare kernel calls):     ~1209.9 ms / block
  speedup: ~1.11x ; all 4 metric CIs BIT-IDENTICAL.
"""
from __future__ import annotations

import time

import numpy as np


def main(n=20000, n_bootstrap=1000, reps=12):
    from mlframe.evaluation.bootstrap import bootstrap_metrics
    from mlframe.metrics.core import fast_brier_score_loss as B, fast_log_loss as L
    from mlframe.calibration.policy import _ece_score

    rng = np.random.default_rng(3)
    raw = rng.uniform(0, 1, n)
    tp = 1.0 / (1.0 + np.exp(-4.0 * (raw - 0.5)))
    y = (rng.uniform(0, 1, n) < tp).astype(np.int64)
    p = raw

    def br_old(yy, pp):
        return float(B(yy.astype(np.float64, copy=False), pp.astype(np.float64, copy=False)))

    def ll_old(yy, pp):
        return float(L(yy.astype(np.float64, copy=False), pp.astype(np.float64, copy=False)))

    def br_new(yy, pp):
        return float(B(yy, pp))

    def ll_new(yy, pp):
        return float(L(yy, pp))

    ece = lambda yy, pp: _ece_score(yy, pp)
    mf_old = {"brier": br_old, "log_loss": ll_old, "ece": ece}
    mf_new = {"brier": br_new, "log_loss": ll_new, "ece": ece}
    yf = np.ascontiguousarray(y, dtype=np.float64)
    pf = np.ascontiguousarray(p, dtype=np.float64)

    bootstrap_metrics(y, p, mf_old, n_bootstrap=100, alpha=0.05, stratify=y, random_state=42)
    bootstrap_metrics(yf, pf, mf_new, n_bootstrap=100, alpha=0.05, stratify=y, random_state=42)

    co = bootstrap_metrics(y, p, mf_old, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y, random_state=42)
    cn = bootstrap_metrics(yf, pf, mf_new, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y, random_state=42)
    identical = all(co[m]["point"] == cn[m]["point"] and co[m]["lo"] == cn[m]["lo"] and co[m]["hi"] == cn[m]["hi"] for m in mf_old)

    t = time.perf_counter()
    for _ in range(reps):
        bootstrap_metrics(y, p, mf_old, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y, random_state=42)
    old_ms = (time.perf_counter() - t) / reps * 1000
    t = time.perf_counter()
    for _ in range(reps):
        bootstrap_metrics(yf, pf, mf_new, n_bootstrap=n_bootstrap, alpha=0.05, stratify=y, random_state=42)
    new_ms = (time.perf_counter() - t) / reps * 1000

    print(f"n={n} n_bootstrap={n_bootstrap} reps={reps}")
    print(f"old={old_ms:.2f}ms new={new_ms:.2f}ms speedup={old_ms / new_ms:.3f}x bit_identical={identical}")


if __name__ == "__main__":
    main()
