"""Bench (lead4): shared resample-index reuse across calibrator candidates.

``pick_best_calibrator`` benches ~3-4 candidates; the OLD path called
``bootstrap_metric`` per candidate, which regenerated the identical stratified
resample index matrix (same n / stratify / seed) every time. The NEW path builds
the index matrix ONCE via ``_build_resample_indices`` and reuses it across all
candidates through ``_bootstrap_ece_with_indices`` -- bit-identical CIs (same
indices) at lower cost.

Run: python -m mlframe.calibration._benchmarks.bench_pick_best_calibrator_index_reuse

Measured (n=2000 OOF, n_bootstrap=500, 4 candidates, warm, 20 reps, dev box):
  old (per-candidate bootstrap_metric regen):  ~89.2 ms / call
  new (single shared index matrix):            ~53.4 ms / call
  speedup: ~1.67x ; chosen calibrator + every alternative CI BIT-IDENTICAL.
"""
from __future__ import annotations

import time

import numpy as np


def _make(n, seed=11):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, n)
    tp = 1.0 / (1.0 + np.exp(-6.0 * (raw - 0.5)))
    y = (rng.uniform(0, 1, n) < tp).astype(np.int64)
    return raw, y


def main(n=2000, n_bootstrap=500, reps=20):
    from mlframe.calibration import policy
    from mlframe.calibration.policy import pick_best_calibrator
    from mlframe.evaluation.bootstrap import bootstrap_metric

    raw, y = _make(n)
    strat = y if np.unique(y).size == 2 else None

    pick_best_calibrator(probs=None, y=None, oof_probs=raw, oof_y=y, n_bootstrap=200, random_state=11)

    t = time.perf_counter()
    for _ in range(reps):
        out_new = pick_best_calibrator(probs=None, y=None, oof_probs=raw, oof_y=y, n_bootstrap=n_bootstrap, random_state=11)
    new_ms = (time.perf_counter() - t) / reps * 1000

    orig = policy._bootstrap_ece_with_indices

    def legacy(yt, yp, idx, mf, alpha):
        ci = bootstrap_metric(yt, yp, metric_fn=mf, n_bootstrap=n_bootstrap, alpha=alpha, stratify=strat, random_state=11)
        return {"point": ci["point"], "lo": ci["lo"], "hi": ci["hi"]}

    policy._bootstrap_ece_with_indices = legacy
    pick_best_calibrator(probs=None, y=None, oof_probs=raw, oof_y=y, n_bootstrap=200, random_state=11)
    t = time.perf_counter()
    for _ in range(reps):
        out_old = pick_best_calibrator(probs=None, y=None, oof_probs=raw, oof_y=y, n_bootstrap=n_bootstrap, random_state=11)
    old_ms = (time.perf_counter() - t) / reps * 1000
    policy._bootstrap_ece_with_indices = orig

    identical = out_new["chosen"] == out_old["chosen"] and all(
        out_new["alternatives"][k]["ece_ci"] == out_old["alternatives"][k]["ece_ci"]
        for k in out_new["alternatives"]
    )
    print(f"n={n} n_bootstrap={n_bootstrap} reps={reps}")
    print(f"old={old_ms:.2f}ms new={new_ms:.2f}ms speedup={old_ms / new_ms:.2f}x")
    print(f"chosen={out_new['chosen']} bit_identical={identical}")


if __name__ == "__main__":
    main()
