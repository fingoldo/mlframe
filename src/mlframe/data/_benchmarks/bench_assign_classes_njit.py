"""Bench: assign_classes_from_probability pure-Python double loop vs njit kernel.

OLD = the exact prior per-row Python loop (reproduced inline below).
NEW = mlframe.data.synthetic._assign_classes_from_probability_kernel (njit, cache=True).

The per-row cumulative-probability walk is order-invariant and integer-valued, so the
njit result is BIT-IDENTICAL to the Python loop (exact ==), not merely close.

Run:
    python -m mlframe.data._benchmarks.bench_assign_classes_njit
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.data.synthetic import _assign_classes_from_probability_kernel


def _old_python(predictors, draw, n_classes, out):
    n_samples = predictors.shape[0]
    for i in range(n_samples):
        total = 0.0
        out[i] = n_classes - 1
        for j in range(n_classes):
            total += predictors[i, j]
            if draw[i] < total:
                out[i] = j
                break
    return out


def _make(n_samples, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random((n_samples, n_classes)).astype(np.float32)
    p /= p.sum(axis=1, keepdims=True)
    draw = rng.random(n_samples)
    return p, draw


def _best_of(fn, *args, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        out = np.empty(args[0].shape[0], dtype=np.int32)
        t = time.perf_counter()
        fn(*args, out)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    # warm njit
    p0, d0 = _make(64, 3)
    _assign_classes_from_probability_kernel(p0, d0, 3, np.empty(64, dtype=np.int32))

    print(f"{'n_samples':>10} {'n_cls':>5} {'OLD ms':>10} {'NEW ms':>10} {'speedup':>8} {'identical':>10}")
    for n_samples in (10_000, 100_000, 1_000_000):
        for n_classes in (3, 8):
            p, draw = _make(n_samples, n_classes)

            old_out = np.empty(n_samples, dtype=np.int32)
            _old_python(p, draw, n_classes, old_out)
            new_out = np.empty(n_samples, dtype=np.int32)
            _assign_classes_from_probability_kernel(p, draw, n_classes, new_out)
            identical = np.array_equal(old_out, new_out)

            t_old = _best_of(_old_python, p, draw, n_classes)
            t_new = _best_of(_assign_classes_from_probability_kernel, p, draw, n_classes)
            print(f"{n_samples:>10} {n_classes:>5} {t_old*1e3:>10.3f} {t_new*1e3:>10.3f} {t_old/t_new:>7.1f}x {str(identical):>10}")


if __name__ == "__main__":
    main()
