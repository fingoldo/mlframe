"""Bench whether the per-class loop in
``compute_probabilistic_multiclass_error`` benefits from threading.

Each iteration calls fast_ice_only / fast_brier_score_loss / fast_aucs.
These kernels already release the GIL (nogil=True) and the >100k-row
ones already auto-dispatch to parallel internally.

ThreadPoolExecutor across classes COULD let multiple class-iterations
run concurrently, since the inner kernels release the GIL. But there's
overhead: each Python class-iteration has to validate / coerce its inputs,
and we'd need to merge K results back at the end.

Decision rule: ship a threaded variant if speedup >= 2x at K=10, N=200k.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, ".")

from mlframe.metrics.core import compute_probabilistic_multiclass_error  # noqa: E402


def threaded_compute(y_true, y_score, **kwargs):
    """Each class computed concurrently by the executor.

    Reuses the existing single-class fast path by calling the public
    function with a single column at a time, then weighted-aggregating
    at the end.
    """
    if y_score.ndim == 1:
        y_score = np.vstack([1 - y_score, y_score]).T
    K = y_score.shape[1]
    if K <= 2:
        # Threading overhead dominates for binary; just call once.
        return compute_probabilistic_multiclass_error(y_true, y_score, **kwargs)

    def _one_class(class_id):
        # Hack: call the full function but slicing to a single class.
        # In practice we'd need to refactor compute_probabilistic_multiclass_error
        # to expose its per-class kernel; this proxies the per-class cost.
        single_score = np.vstack([1 - y_score[:, class_id], y_score[:, class_id]]).T
        return compute_probabilistic_multiclass_error(
            (y_true == class_id).astype(np.int8), single_score, **kwargs,
        )

    with ThreadPoolExecutor(max_workers=min(K, 8)) as ex:
        per_class = list(ex.map(_one_class, range(K)))
    return float(np.mean(per_class))


def time_op(fn, *args, repeats=3, warmup=1, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        t.append(time.perf_counter() - t0)
    return out, min(t)


def fmt(t):
    if t < 1e-3:
        return f"{t*1e6:7.1f}us"
    if t < 1.0:
        return f"{t*1e3:7.2f}ms"
    return f"{t:7.3f}s"


def main():
    rng = np.random.default_rng(0)
    print(f"{'N':>9} {'K':>3} | {'seq':>10} | {'threaded':>10} | {'par/seq':>8}")
    print("-" * 55)
    for N, K in [(10_000, 3), (100_000, 3), (200_000, 3),
                 (200_000, 5), (200_000, 10), (1_000_000, 5)]:
        y_true = rng.integers(0, K, size=N)
        # K-column probability matrix
        logits = rng.standard_normal((N, K))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        y_score = e / e.sum(axis=1, keepdims=True)

        compute_probabilistic_multiclass_error(y_true, y_score, method="brier_score")  # warm
        threaded_compute(y_true, y_score, method="brier_score")

        seq_v, t_seq = time_op(
            compute_probabilistic_multiclass_error, y_true, y_score,
            method="brier_score", repeats=3,
        )
        par_v, t_par = time_op(threaded_compute, y_true, y_score,
                                method="brier_score", repeats=3)
        print(f"{N:>9} {K:>3} | {fmt(t_seq):>10} | {fmt(t_par):>10} | {t_par/t_seq:7.2f}x")


if __name__ == "__main__":
    main()
