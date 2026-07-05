"""Bench: hot-path levers for ``_ranker_fs.group_aware_relevance`` / ``_binned_mi``.

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training.ranking._benchmarks.bench_binned_mi_group_relevance

Context (2026-06-23 optimization wave):
``group_aware_relevance`` calls ``_binned_mi`` ``n_features * n_groups`` times. On the
realistic LTR shape (n=50k rows, G=2000 query groups, F=8 features) the full function
is ~13.3s (cProfile) / ~41.5s (wall at F=40). cProfile (sort=tottime) attributes the
cost to ``np.quantile`` (2 calls / ``_binned_mi``, 32000 total, ~5.2s cumtime) +
``np.unique`` on the quantile edges + the generic ``_quantile`` machinery (partition /
lerp / gamma / clip / indexes) -- i.e. per-call NumPy dispatch overhead on TINY arrays
(query groups average ~25 rows), NOT the histogram accumulation.

Two bit-identical levers were measured here:

1. mask/size build in ``group_aware_relevance`` -- compute ``sizes`` from ``m.sum()``
   over the already-built masks instead of a second ``groups == g`` equality pass.
   ISOLATED 1.26x on the sub-step (282ms -> 224ms at G=2000), bit-identical. But the
   sub-step is ~0.7% of the full function, so e2e gain ~0.14% -- REJECTED (below ~5%).

2. ``np.add.at(joint, (xb, yb), 1.0)`` -> ``np.bincount(xb*ny+yb)`` joint histogram.
   bincount beats add.at only at large per-group n (1.05x @ n=25, 1.15x @ n=100,
   2.66x @ n=1000). At the REAL call shape (16000 calls, groups 8-40 rows) the
   bincount alloc+reshape overhead dominates: measured 0.979x (SLOWER) e2e on the
   small-group mix. REJECTED, measured e2e.

The only lever that would clear the bar is to ``@njit`` the whole ``_binned_mi``
(removing the per-call quantile dispatch), but that requires a manual
linear-interpolation quantile whose edges diverge from ``np.quantile`` by ~1e-16
(single-ULP FP order). Those edges feed ``searchsorted`` bin assignment, so a tie at
an edge can flip a sample's bin and move the MI by more than 1e-9 -- a selection-
altering risk in a feature SELECTOR (the "rebinning is suspect on tied data" class).
That fails the identity gate for this wave; deferred (FUTURE) until a provably
edge-exact njit quantile or a selection-equivalence validation harness exists.
"""
from __future__ import annotations

import time

import numpy as np


def _binned_mi_addat(x, y, bins=8):
    n = x.shape[0]
    if n < 4:
        return 0.0
    fin = np.isfinite(x) & np.isfinite(y)
    if int(fin.sum()) < 4:
        return 0.0
    x, y = x[fin], y[fin]
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0
    xe = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    ye = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    if xe.size < 2 or ye.size < 2:
        return 0.0
    xb = np.clip(np.searchsorted(xe[1:-1], x, side="right"), 0, xe.size - 2)
    yb = np.clip(np.searchsorted(ye[1:-1], y, side="right"), 0, ye.size - 2)
    joint = np.zeros((xe.size - 1, ye.size - 1), dtype=np.float64)
    np.add.at(joint, (xb, yb), 1.0)
    joint /= x.shape[0]
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))


def _binned_mi_bincount(x, y, bins=8):
    n = x.shape[0]
    if n < 4:
        return 0.0
    fin = np.isfinite(x) & np.isfinite(y)
    if int(fin.sum()) < 4:
        return 0.0
    x, y = x[fin], y[fin]
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0
    xe = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    ye = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    if xe.size < 2 or ye.size < 2:
        return 0.0
    nx, ny = xe.size - 1, ye.size - 1
    xb = np.clip(np.searchsorted(xe[1:-1], x, side="right"), 0, nx - 1)
    yb = np.clip(np.searchsorted(ye[1:-1], y, side="right"), 0, ny - 1)
    joint = np.bincount(xb * ny + yb, minlength=nx * ny).astype(np.float64).reshape(nx, ny)
    joint /= x.shape[0]
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))


def _bench_mask_build(seed=0, n=50000, g=2000, it=15):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, g, size=n)
    uniq = np.unique(groups)

    def old(gr, uq):
        sizes = np.array([int(np.sum(gr == c)) for c in uq], dtype=np.float64)
        masks = [gr == c for c in uq]
        return sizes, masks

    def new(gr, uq):
        masks = [gr == c for c in uq]
        sizes = np.array([int(m.sum()) for m in masks], dtype=np.float64)
        return sizes, masks

    old(groups, uniq)
    new(groups, uniq)

    def best(fn):
        return min(_time(lambda: fn(groups, uniq)) for _ in range(it))

    to, tn = best(old), best(new)
    so, _ = old(groups, uniq)
    sn, _ = new(groups, uniq)
    print(f"[mask-build] OLD {to*1000:.2f}ms NEW {tn*1000:.2f}ms speedup {to/tn:.2f}x " f"identical={np.array_equal(so, sn)}  (REJECT: ~0.7% of full fn)")


def _bench_joint_hist(seed=0):
    rng = np.random.default_rng(seed)
    calls = []
    for _ in range(16000):
        nn = int(rng.integers(8, 40))
        calls.append((rng.standard_normal(nn), rng.standard_normal(nn)))
    maxd = max(abs(_binned_mi_addat(x, y) - _binned_mi_bincount(x, y)) for x, y in calls)

    def run(fn):
        s = time.perf_counter()
        for x, y in calls:
            fn(x, y)
        return time.perf_counter() - s

    run(_binned_mi_addat)
    run(_binned_mi_bincount)
    to = min(run(_binned_mi_addat) for _ in range(3))
    tn = min(run(_binned_mi_bincount) for _ in range(3))
    print(f"[joint-hist add.at->bincount, real small-group mix] OLD {to*1000:.1f}ms "
          f"NEW {tn*1000:.1f}ms speedup {to/tn:.3f}x max_abs_diff={maxd:g}  "
          f"(REJECT: <1x at the real group sizes)")


def _time(fn):
    s = time.perf_counter()
    fn()
    return time.perf_counter() - s


if __name__ == "__main__":
    _bench_mask_build()
    _bench_joint_hist()
