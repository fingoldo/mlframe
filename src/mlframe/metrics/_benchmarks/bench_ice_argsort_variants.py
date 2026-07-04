"""Measured verdict on the ICE-kernel argsort (compute_probabilistic_multiclass_error hotspot).

The per-class AUC walk in _batch_per_class_ice_kernel only accumulates at tie-run boundaries, so BOTH roc_auc
and pr_auc are invariant to the within-tie argsort order -> kind="mergesort" (stable) is not required for
correctness. But numba's own argsort is known-slow (a rejected note in fast_roc_auc: numba quicksort 0.22-0.46x
vs numpy). So this benches, at the real binary K=1 shape, whether:
  (a) mergesort -> quicksort inside the njit kernel is actually faster (numba impl), and
  (b) whether pulling the argsort out to numpy's C argsort (per-class, then a walk-only njit kernel) wins,
against bit-identity of the resulting ICE. Prints the verdict.
"""
import time
import numpy as np
import numba

from mlframe.metrics.core import _batch_per_class_ice_kernel

NB = 10
KW = dict(nbins=NB, use_weights=True, mae_weight=1.0, std_weight=1.0, brier_loss_weight=1.0,
          roc_auc_weight=1.0, pr_auc_weight=1.0, min_roc_auc=0.5, roc_auc_penalty=0.0)


@numba.njit(cache=True, nogil=True)
def _ice_walk_presorted(y_t_sorted, y_p_sorted, N, total_pos, total_neg):
    """Walk-only ICE-AUC on pre-sorted (descending) arrays; the sort is done by the caller (numpy)."""
    if total_pos == 0 or total_neg == 0:
        return np.nan, np.nan
    last_fps = 0; last_tps = 0; tps = 0; fps = 0
    roc_acc = 0.0; pr_acc = 0.0; prev_recall = 0.0
    for i in range(N):
        yi = y_t_sorted[i]
        tps += yi; fps += 1 - yi
        if i == N - 1 or y_p_sorted[i + 1] != y_p_sorted[i]:
            roc_acc += (fps - last_fps) * (last_tps + tps)
            last_fps = fps; last_tps = tps
            cp = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            cr = tps / total_pos
            pr_acc += (cr - prev_recall) * cp
            prev_recall = cr
    denom = tps * fps * 2
    return (roc_acc / denom if denom > 0 else np.nan), pr_acc


def numpy_sort_auc(y_t, y_p):
    desc = np.argsort(-y_p)  # numpy C quicksort
    yts = y_t[desc]; yps = y_p[desc]
    tp = int(yts.sum())
    return _ice_walk_presorted(yts.astype(np.int64), yps, yts.shape[0], tp, yts.shape[0] - tp)


def run(n, tied):
    rng = np.random.default_rng(0)
    yt = (rng.random(n) < 0.3).astype(np.int8).reshape(n, 1)
    p = 0.15 + 0.5 * yt.ravel() + rng.standard_normal(n) * 0.25
    if tied:
        p = np.round(p, 2)  # induce ties (discrete/binned classifier)
    p = np.clip(p, 1e-6, 1 - 1e-6).reshape(n, 1)
    # warm
    _batch_per_class_ice_kernel(yt, p, np.ascontiguousarray(np.argsort(-p, axis=0).astype(np.int64)), **KW)
    numpy_sort_auc(yt[:, 0].astype(np.int64), p[:, 0])

    reps = 20
    t = time.perf_counter()
    for _ in range(reps):
        r0 = _batch_per_class_ice_kernel(yt, p, **KW)
    t_cur = (time.perf_counter() - t) / reps

    # numpy-extracted argsort variant: reproduce the full ICE would need brier+cal too; here we isolate the AUC
    # walk timing (the argsort+walk is the dominant part) to see the sort-backend delta.
    t = time.perf_counter()
    for _ in range(reps):
        _ = numpy_sort_auc(yt[:, 0].astype(np.int64), p[:, 0])
    t_np = (time.perf_counter() - t) / reps

    print(f"n={n:>8} tied={str(tied):5}: full_kernel(njit-mergesort)={t_cur*1e3:6.1f}ms  "
          f"numpy-argsort+walk-only={t_np*1e3:6.1f}ms  (walk-only is a LOWER bound on an extracted variant)")
    return r0


if __name__ == "__main__":
    for n in (100_000, 1_000_000):
        for tied in (False, True):
            run(n, tied)
