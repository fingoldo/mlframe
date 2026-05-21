"""Bench compute_mi_from_classes: indexed-loop vs zip-iter.

The current kernel uses ``for i, j in zip(classes_x, classes_y)`` which
in numba @njit creates an iterator pair per iteration -- tuple boxing
may add overhead vs the equivalent ``for k in range(n)`` with manual
``classes_x[k]``, ``classes_y[k]`` reads.

c0013 iter143 profile attributed 0.275 s self-time across 239 calls
(~1.15 ms / call). Even 10% saves 30 ms per fit on this combo; for
combos that run the bandit Phase 2 (which calls this many times),
the savings amortise across many calls.

Run: ``python profiling/bench_compute_mi_from_classes_no_zip.py``
"""

import time
import math
import numpy as np
from numba import njit


@njit(cache=True)
def mi_zip(classes_x, freqs_x, classes_y, freqs_y, dtype):
    joint_counts = np.zeros((len(freqs_x), len(freqs_y)), dtype=dtype)
    for i, j in zip(classes_x, classes_y):
        joint_counts[i, j] += 1
    joint_freqs = joint_counts / len(classes_x)
    total = 0.0
    for i in range(len(freqs_x)):
        prob_x = freqs_x[i]
        for j in range(len(freqs_y)):
            jf = joint_freqs[i, j]
            if jf:
                prob_y = freqs_y[j]
                total += jf * math.log(jf / (prob_x * prob_y))
    return total


@njit(cache=True)
def mi_range(classes_x, freqs_x, classes_y, freqs_y, dtype):
    n = len(classes_x)
    K_x = len(freqs_x)
    K_y = len(freqs_y)
    joint_counts = np.zeros((K_x, K_y), dtype=dtype)
    for k in range(n):
        joint_counts[classes_x[k], classes_y[k]] += 1
    inv_n = 1.0 / n
    total = 0.0
    for i in range(K_x):
        prob_x = freqs_x[i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                # joint_freqs = jc / n; log(jf / (px*py)) preserved
                total += jf * math.log(jf / (prob_x * prob_y))
    return total


def bench(label, fn, args, n_iter=200):
    fn(*args); fn(*args)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(*args)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e6, label


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n, Kx, Ky in [(50_000, 8, 3), (200_000, 8, 3), (1_000_000, 8, 3),
                       (200_000, 16, 5), (1_000_000, 16, 5)]:
        cx = rng.integers(0, Kx, size=n).astype(np.int32)
        cy = rng.integers(0, Ky, size=n).astype(np.int32)
        fx = (np.bincount(cx, minlength=Kx) / n).astype(np.float64)
        fy = (np.bincount(cy, minlength=Ky) / n).astype(np.float64)
        args = (cx, fx, cy, fy, np.int32)
        v_zip = mi_zip(*args)
        v_rng = mi_range(*args)
        t_zip, _ = bench("zip",   mi_zip,   args)
        t_rng, _ = bench("range", mi_range, args)
        print(f"n={n:>7}  K=({Kx},{Ky}): zip={t_zip:7.1f}us  range={t_rng:7.1f}us  ({t_zip/t_rng:.2f}x)  diff={abs(v_zip-v_rng):.2e}")
