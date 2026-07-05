"""Microbench: can ks_statistic SHARE the AUC score-descending argsort?

``ks_statistic`` does its own ascending ``np.argsort(ys)`` then walks the
class-conditional CDFs. The AUC path (``_argsort_desc_for_metrics`` ->
``np.argsort(y_score)[::-1]``) computes a DESCENDING order over the same
scores. The lead asks: feed ONE sort to both (KS asc == reverse of AUC
desc) and save a sort per class.

This bench measures:
  1. KS's own argsort vs the total ks_statistic cost (is the sort the bulk?).
  2. A KS kernel fed a reversed-view of the AUC descending order vs the
     native ascending sort (bit-identity + speed).
  3. The architectural reality: in the report path AUC is computed by the
     BATCHED ``compute_batch_aucs`` (all K cols, possibly GPU) which does
     NOT return its per-column order array -- so there is nothing to share
     without restructuring the batched/GPU kernels.

Run: python -m mlframe.metrics._benchmarks.bench_ks_shared_sort

REJECT (measured): sharing IS bit-identical (KS via reversed AUC-descending
order == native ascending, identical=True at n=10k/100k/500k) and the sort
IS the bulk of ks_statistic (55-125%). BUT not implementable as a micro-
change: in the report the AUC sort lives inside the BATCHED, GPU-capable
compute_batch_aucs, which returns only scalar (roc, pr) per column and never
materializes the per-column order. Threading a (N, K) int64 order matrix
from the batched/GPU AUC kernels back into the per-class KS calls is a
cross-module restructure that also ADDS an 8*N*K allocation -- unacceptable
on a 100GB-frame codebase to save one O(N log N) sort/class. ks_statistic's
own argsort cannot share with anything in the standalone call (numpy-C-bound,
extras-block entanglement). Keeps its independent sort.
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.metrics.classification._classification_extras import (
    _ks_statistic_kernel,
    ks_statistic,
)


def ks_via_desc_reverse(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Feed the KS kernel an ascending order obtained by REVERSING a
    descending argsort (the order the AUC path produces). Tests whether a
    shared AUC-desc sort can drive KS bit-identically."""
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    ys = np.asarray(y_score, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    desc = np.argsort(ys)[::-1]  # what AUC computes
    asc = desc[::-1]  # reverse-view -> ascending
    asc = np.ascontiguousarray(asc)  # kernel walks it; make contiguous
    return float(_ks_statistic_kernel(yt[asc], ys[asc]))


def _time(fn, *args, iters: int = 100) -> float:
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    rng = np.random.default_rng(1)
    for n in (10_000, 100_000, 500_000):
        ys = rng.random(n)
        yt = (rng.random(n) < 0.3).astype(np.int64)

        base = ks_statistic(yt, ys)
        shared = ks_via_desc_reverse(yt, ys)
        identical = base == shared

        # cost of JUST the argsort vs full ks_statistic
        t_sort = _time(lambda a: np.argsort(a, kind="quicksort"), ys)
        t_ks = _time(ks_statistic, yt, ys)
        t_shared = _time(ks_via_desc_reverse, yt, ys)
        print(
            f"n={n:>7}  argsort_alone={t_sort:8.1f}us  ks_full={t_ks:8.1f}us  "
            f"(sort={100*t_sort/t_ks:4.0f}% of ks)  ks_desc_reverse={t_shared:8.1f}us  "
            f"identical={identical}"
        )

    print(
        "\nNote: even a FREE shared sort caps the win at the sort fraction of "
        "ks_full; the report's AUC sort lives inside the BATCHED compute_batch_aucs "
        "(K cols at once, GPU-capable) and is never materialized for KS to reuse."
    )


if __name__ == "__main__":
    main()
