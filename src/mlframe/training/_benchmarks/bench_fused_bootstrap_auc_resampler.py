"""Discovery microbench for the NEW top hotspot in the bootstrap-metrics path.

After wave-7 (fast_roc_auc pre-argsort, calibrator resample reuse, _bootstrap
precast), re-profiling PREDICT + honest_diagnostics + bootstrap puts the dominant
mlframe-side cost on the AUC resampler closure ``_resampler_fast`` in
``metrics/_core_auc_brier.py:178`` -- 16.0s tottime of 33s at n_rows=300k, 1001
calls. Its per-call work is THREE separate njit/Python boundaries plus a duplicate
fancy-index gather:

  1. ``base_rank[idx]``                 (n int64 gather)
  2. ``_resample_desc_order_counting``  (njit O(n) counting sort + ``[::-1]``)
  3. ``y_true[idx]`` / ``y_score[idx]`` (TWO more n gathers -- duplicated: the
     bootstrap_metrics loop already computed yt=y_true[idx], yp=y_pred[idx])
  4. ``fast_numba_auc_nonw``            (njit; re-gathers y/score by desc indices)

This bench proves component costs are REAL (not cProfile attribution noise) and
prototypes a single fused njit kernel that:
  - takes only ``idx`` + a precomputed ``base_rank`` + ``y_by_rank`` (y reindexed
    into ascending-score-rank order, built ONCE in the factory),
  - bins counts AND positive-counts per rank in one pass,
  - walks ranks descending accumulating tps/fps directly -- NO desc index array,
    NO ``[::-1]``, NO second/third gather, ONE njit boundary.

Bit-identical to ``fast_roc_auc_unstable(y[idx], y_score[idx])`` on tie-free
float64 base scores (same all-distinct gate wave-7 already uses).

SHIPPED: the fused kernel is now ``_fused_resample_auc`` in
``metrics/_core_auc_brier.py`` and is the default fast path of
``make_bootstrap_auc_resampler`` (gated on all-distinct base scores; tied/discrete
falls back to the exact argsort path). Measured warm on this host:
    n=50_000:  current=0.717s fused=0.416s  1.72x  maxdiff 0.0
    n=200_000: current=8.606s fused=3.990s  2.16x  maxdiff 0.0

Run (CPU-only):
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.bench_fused_bootstrap_auc_resampler
"""
from __future__ import annotations

import os
import time

import numba
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from mlframe.metrics._numba_params import NUMBA_NJIT_PARAMS


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fused_resample_auc(idx, base_rank, y_by_rank, n):
    """Single-pass resample AUC: bin counts+positives by ascending rank, then
    walk descending. No desc-index array, no duplicate y/score gather."""
    counts = np.zeros(n, dtype=np.int64)
    ones = np.zeros(n, dtype=np.int64)
    m = idx.shape[0]
    for k in range(m):
        r = base_rank[idx[k]]
        counts[r] += 1
        ones[r] += y_by_rank[r]
    last_fps = 0
    last_tps = 0
    tps = 0
    fps = 0
    auc = 0
    for r in range(n - 1, -1, -1):
        c = counts[r]
        if c == 0:
            continue
        pos = ones[r]
        neg = c - pos
        tps += pos
        fps += neg
        auc += (fps - last_fps) * (last_tps + tps)
        last_fps = fps
        last_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return auc / tmp
    return np.nan


def main(reps: int = 1000) -> dict:
    from mlframe.metrics._core_auc_brier import (
        make_bootstrap_auc_resampler,
        fast_roc_auc_unstable,
    )

    out = {}
    for n in (50_000, 200_000):
        rng = np.random.default_rng(0)
        y = (rng.random(n) < 0.35).astype(np.int64)
        p = 1.0 / (1.0 + np.exp(-(1.2 * y + rng.normal(0, 1, n))))
        asc = np.argsort(p)
        base_rank = np.empty(n, dtype=np.int64)
        base_rank[asc] = np.arange(n, dtype=np.int64)
        y_by_rank = np.empty(n, dtype=np.int64)
        y_by_rank[base_rank] = y

        idxs = [rng.integers(0, n, size=n, dtype=np.int64) for _ in range(50)]
        cur = make_bootstrap_auc_resampler(y, p)
        cur(idxs[0])
        _fused_resample_auc(idxs[0], base_rank, y_by_rank, n)

        def bench(f):
            its = idxs * ((reps // len(idxs)) + 1)
            t = time.perf_counter()
            for k in range(reps):
                f(its[k])
            return round(time.perf_counter() - t, 3)

        cur_t = bench(cur)
        fused_t = bench(lambda ix: _fused_resample_auc(ix, base_rank, y_by_rank, n))

        max_abs = 0.0
        for ix in idxs[:10]:
            a = _fused_resample_auc(ix, base_rank, y_by_rank, n)
            b = fast_roc_auc_unstable(y[ix], p[ix])
            max_abs = max(max_abs, abs(a - b))
        out[f"n{n}"] = {
            "current_resampler_x{}".format(reps): cur_t,
            "fused_x{}".format(reps): fused_t,
            "speedup": round(cur_t / fused_t, 2) if fused_t else None,
            "max_abs_diff_vs_exact": max_abs,
        }
        print(f"n={n:_}: current={cur_t}s fused={fused_t}s "
              f"speedup={out[f'n{n}']['speedup']}x maxdiff={max_abs:.2e}", flush=True)
    return out


if __name__ == "__main__":
    main()
