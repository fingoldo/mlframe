"""A/B: batched cross-column hinge-breakpoint precheck vs the original per-column detector loop
(2026-07-16, cProfile-driven -- see _hinge_detect_gpu_resident_batch.py's module docstring for the full
mechanism). Finds the K (surviving-column count) crossover: at K=1 the batch call's one-time setup
(upload the (n,K) matrix, one cp.quantile(axis=0)) costs more than it saves, so the per-column detector
wins; at K>=2 the batched precheck's O(1) host<->device round trips (vs K separate ~8-sync calls) wins
by a growing margin. generate_hinge_features's dispatcher (_detect_hinge_breakpoints_for_columns) reads
the measured threshold via MLFRAME_HINGE_BATCH_MIN_K (default 2, from this sweep).

Run: python bench_hinge_batch_vs_percolumn.py
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters._hinge_detect_gpu_resident import detect_hinge_breakpoints_gpu
from mlframe.feature_selection.filters._hinge_detect_gpu_resident_batch import detect_hinge_breakpoints_gpu_batch

N = 99401
KW = dict(
    max_breakpoints=2, min_heldout_r2_uplift=0.01, precheck_qs=(0.30, 0.50, 0.70),
    precheck_min_sse_drop=0.005, cand_q_lo=0.10, cand_q_hi=0.90, n_candidates=24,
    min_rows=200, min_seg_rows=30,
)
K_SWEEP = (1, 2, 4, 8, 16, 32, 64, 216)


def _make_cols(rng, y, k):
    cols = []
    for i in range(k):
        x = rng.standard_normal(N)
        if i % 8 == 0:  # a fraction of columns carry a genuine kink so both paths do real detection work
            tau = rng.uniform(-1, 1)
            x = x + np.maximum(x - tau, 0) * 3.0 + 0.3 * y
        cols.append(x)
    return cols


if __name__ == "__main__":
    import cupy as cp

    rng = np.random.default_rng(2)
    y = rng.standard_normal(N)

    detect_hinge_breakpoints_gpu_batch(_make_cols(rng, y, 2), y, **KW)
    detect_hinge_breakpoints_gpu(_make_cols(rng, y, 1)[0], y, **KW)
    cp.cuda.Stream.null.synchronize()

    print(f"{'K':>5}  {'batch_ms':>10}  {'percol_ms':>10}  {'speedup':>8}  winner")
    for k in K_SWEEP:
        cols = _make_cols(rng, y, k)

        t0 = time.perf_counter()
        batch_out = detect_hinge_breakpoints_gpu_batch(cols, y, **KW)
        cp.cuda.Stream.null.synchronize()
        t_batch = time.perf_counter() - t0

        t0 = time.perf_counter()
        percol_out = [detect_hinge_breakpoints_gpu(x, y, **KW) for x in cols]
        cp.cuda.Stream.null.synchronize()
        t_percol = time.perf_counter() - t0

        mismatches = sum(
            1 for b, r in zip(batch_out, percol_out)
            if sorted(round(t, 6) for t in (b or [])) != sorted(round(t, 6) for t in (r or []))
        )
        winner = "batch" if t_batch < t_percol else "per-col"
        print(f"{k:5d}  {t_batch*1000:10.1f}  {t_percol*1000:10.1f}  {t_percol/t_batch:8.2f}x  {winner}  mismatches={mismatches}")
