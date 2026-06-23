"""CPX27 bench: incremental sorted buffer vs per-step ``np.sort`` for ACI radius.

The online ACI controller computes a per-step rolling-quantile radius. The OLD
path did a full ``np.sort`` of the ``buffer_n``-element FIFO window EVERY step
(O(W log W) per step). The NEW path keeps the buffer sorted incrementally with
``bisect.insort`` + a ``bisect``-located eviction so the quantile is an O(1)
index lookup.

Driver: many sequential single-residual ACI steps over a fixed window. Reports
warm best-of-N wall time for OLD (a vendored copy of the pre-change kernel) and
NEW (the live module), plus per-step radius identity over the whole run.

Run::

    CUDA_VISIBLE_DEVICES="" python src/mlframe/training/composite/_benchmarks/bench_cpx27_conformal_online.py
"""
from __future__ import annotations

import bisect
import math
import time

import numpy as np


# ---- OLD kernel (vendored pre-CPX27: full np.sort per step) ----------------

def _old_radius(residuals, alpha):
    r = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
    r = r[np.isfinite(r)]
    m = int(r.size)
    if m == 0:
        return float("inf")
    if alpha <= 0.0:
        return float("inf")
    if alpha >= 1.0:
        return 0.0
    rank = int(math.ceil((m + 1) * (1.0 - alpha)))
    if rank > m:
        return float("inf")
    r_sorted = np.sort(r)
    return float(r_sorted[rank - 1])


def _run_old(residuals, alphas, buffer_n):
    buf = []
    out = []
    for ar, alpha in zip(residuals, alphas):
        # radius is read pre-append (online contract: current-as-of-row)
        out.append(_old_radius(np.asarray(buf, dtype=np.float64), alpha) if buf else float("inf"))
        buf.append(ar)
        if len(buf) > buffer_n:
            del buf[: len(buf) - buffer_n]
    return out


# ---- NEW kernel (incremental sorted buffer) --------------------------------

def _new_radius(r_sorted, m, alpha):
    if m == 0:
        return float("inf")
    if alpha <= 0.0:
        return float("inf")
    if alpha >= 1.0:
        return 0.0
    rank = int(math.ceil((m + 1) * (1.0 - alpha)))
    if rank > m:
        return float("inf")
    return float(r_sorted[rank - 1])


def _run_new(residuals, alphas, buffer_n):
    buf = []
    buf_sorted = []
    out = []
    for ar, alpha in zip(residuals, alphas):
        out.append(_new_radius(buf_sorted, len(buf_sorted), alpha))
        buf.append(ar)
        bisect.insort(buf_sorted, ar)
        if len(buf) > buffer_n:
            n_evict = len(buf) - buffer_n
            for evicted in buf[:n_evict]:
                del buf_sorted[bisect.bisect_left(buf_sorted, evicted)]
            del buf[:n_evict]
    return out


def _bench(fn, *args, n=5):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    rng = np.random.default_rng(20260623)
    for steps, buffer_n in [(50_000, 1000), (50_000, 5000)]:
        residuals = np.abs(rng.normal(size=steps)).tolist()
        # alpha_t wandering in (0,1) like the controller drives it
        alphas = np.clip(0.1 + 0.05 * np.cumsum(rng.normal(scale=0.01, size=steps)), 0.001, 0.999).tolist()

        # warm
        _run_old(residuals[:2000], alphas[:2000], buffer_n)
        _run_new(residuals[:2000], alphas[:2000], buffer_n)

        t_old, out_old = _bench(_run_old, residuals, alphas, buffer_n)
        t_new, out_new = _bench(_run_new, residuals, alphas, buffer_n)

        identical = all(
            (a == b) or (math.isinf(a) and math.isinf(b))
            for a, b in zip(out_old, out_new)
        )
        print(f"steps={steps} window={buffer_n}: OLD={t_old*1e3:8.1f}ms  NEW={t_new*1e3:8.1f}ms"
              f"  speedup={t_old/t_new:5.2f}x  identical={identical}")


if __name__ == "__main__":
    main()
