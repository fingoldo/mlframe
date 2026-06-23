"""Bench for CPX17: the flavour-INVARIANT outlier-member gate (cross-member median +
per-member MAE/STD) is recomputed once per flavour inside ``ensemble_probabilistic_predictions``.

``score_ensemble`` fans ``_process_single_ensemble_method`` over n_flavours; each flavour's
per-split call re-runs ``np.median(axis=0)`` + ``_per_member_mae_std`` + the threshold gate on
the SAME member array. The gate result (kept-member mask) does not depend on the flavour.

This bench measures:
  1. the isolated cost of the gate (median + per-member MAE/STD) at a realistic split shape;
  2. the end-to-end cost of running n_flavours x ensemble_probabilistic_predictions on one split
     OLD (gate recomputed per flavour) vs NEW (gate computed once, mask threaded through).

Run:
    python src/mlframe/models/ensembling/_benchmarks/bench_score_ensemble.py
"""
from __future__ import annotations

import time
import numpy as np

from mlframe.models.ensembling.predict import ensemble_probabilistic_predictions, _clear_gate_cache
from mlframe.models.ensembling.base import _per_member_mae_std


def _best_of(fn, n=7):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(min(ts))


def main():
    rng = np.random.default_rng(0)
    n_samples = 50_000
    n_members = 16
    n_classes = 2
    flavours = ["arithm", "harm", "median", "quad", "geo"]  # several flavours

    # Realistic member probs: most members clustered, one clear outlier (5x off) so the gate
    # actually drops a member (exercises the re-materialise path too).
    base = rng.uniform(0.0, 1.0, size=(n_samples, n_classes))
    members = []
    for m in range(n_members):
        noise = rng.normal(0.0, 0.02, size=base.shape)
        members.append(np.clip(base + noise, 0.0, 1.0))
    members[0] = np.clip(base + rng.normal(0.0, 0.20, size=base.shape), 0.0, 1.0)  # outlier
    preds = [m.astype(np.float64) for m in members]
    arr = np.asarray(preds, dtype=np.float64)

    # warm numba kernel in _per_member_mae_std
    med = np.median(arr, axis=0)
    _per_member_mae_std(arr, med)

    # 1) isolated gate cost
    def gate_only():
        m = np.median(arr, axis=0)
        _per_member_mae_std(arr, m)

    g_med, g_min = _best_of(gate_only, n=11)
    print(f"[gate-only] median(axis=0)+_per_member_mae_std : median={g_med*1e3:.3f} ms  min={g_min*1e3:.3f} ms")

    # 2a) OLD behaviour: gate recomputed every flavour (simulate by clearing the memo before each call).
    def old_all_flavours():
        out = []
        for fl in flavours:
            _clear_gate_cache()
            r, _, _ = ensemble_probabilistic_predictions(*preds, ensemble_method=fl, verbose=False)
            out.append(r)
        return out

    # 2b) NEW behaviour: gate computed once per split, reused across flavours (cache warm across the loop).
    def new_all_flavours():
        _clear_gate_cache()
        out = []
        for fl in flavours:
            r, _, _ = ensemble_probabilistic_predictions(*preds, ensemble_method=fl, verbose=False)
            out.append(r)
        return out

    old_med, old_min = _best_of(old_all_flavours, n=7)
    new_med, new_min = _best_of(new_all_flavours, n=7)
    print(f"[OLD e2e] {len(flavours)} flavours, gate recomputed per flavour: median={old_med*1e3:.3f} ms  min={old_min*1e3:.3f} ms")
    print(f"[NEW e2e] {len(flavours)} flavours, gate memoised once per split: median={new_med*1e3:.3f} ms  min={new_min*1e3:.3f} ms")
    print(f"   speedup median={old_med/new_med:.2f}x   min={old_min/new_min:.2f}x")
    print(f"   gate share of OLD e2e ~ {len(flavours)*g_med/old_med*100:.1f}% (redundant recomputes saved = {len(flavours)-1} of {len(flavours)})")

    # identity gate: NEW per-flavour outputs must be bit-identical to OLD.
    old_out = old_all_flavours()
    new_out = new_all_flavours()
    ok = all(np.array_equal(a, b) for a, b in zip(old_out, new_out))
    maxabs = max(float(np.max(np.abs(a - b))) for a, b in zip(old_out, new_out))
    print(f"\n[identity] OLD vs NEW per-flavour ensemble outputs: array_equal_all={ok}  max|diff|={maxabs:.3e}")

    print(f"\nshape: n_samples={n_samples} n_members={n_members} n_classes={n_classes} n_flavours={len(flavours)}")


if __name__ == "__main__":
    main()
