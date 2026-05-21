"""Bench indexed range-loop vs zip-iter across the 4 remaining @njit zip sites.

bench-attempt-rejected (2026-05-21, iter144 sweep audit after iter143):
iter143 saw 24-26% from this pattern in compute_mi_from_classes, but
the remaining 4 @njit zip sites benefit only 0-7%:

    fast_calibration_binning:
      n=  50000: zip= 183.0us  range= 182.3us  (1.00x)
      n= 200000: zip= 753.1us  range= 736.7us  (1.02x)
      n=1000000: zip=4059.6us  range=3778.9us  (1.07x)

    fast_precision:
      n=  50000: zip= 173.6us  range= 166.1us  (1.05x)
      n= 200000: zip= 705.3us  range= 668.5us  (1.06x)
      n=1000000: zip=3849.7us  range=3664.8us  (1.05x)

Cause: iter143's compute_mi_from_classes does only 1 simple op per
iteration (``joint_counts[i, j] += 1``) so the zip-unboxing overhead
dominates inner work proportionally. These kernels do MORE work per
iteration (branches, bounds checks, bin computation), so zip overhead
is a smaller fraction. At n=1M / 7% the absolute gain is 280us / call;
even at hot-path call counts (190+ per fit) that's 50ms / fit -- below
the worth-shipping threshold AND the range form is slightly less
idiomatic than the natural Python zip form.

iter143 had a clean win because its inner work was minimal; these
don't. Documented in-code (in the bench script + a one-liner at each
of the four call sites) per
``feedback_document_failed_optimization_attempts`` so the next agent
doesn't re-do the same audit.

Run: ``python profiling/bench_njit_zip_range_sweep.py``
"""

import time
import numpy as np
from math import floor
from numba import njit


@njit(cache=True)
def calib_zip(y_true, y_pred, nbins):
    pockets_predicted = np.zeros(nbins, dtype=np.int64)
    pockets_true = np.zeros(nbins, dtype=np.int64)
    min_val, max_val = 1.0, 0.0
    for predicted_prob in y_pred:
        if predicted_prob > max_val:
            max_val = predicted_prob
        if predicted_prob < min_val:
            min_val = predicted_prob
    span = max_val - min_val
    if span > 0:
        multiplier = (nbins - 1) / span
        for true_class, predicted_prob in zip(y_true, y_pred):
            ind = floor((predicted_prob - min_val) * multiplier)
            pockets_predicted[ind] += 1
            pockets_true[ind] += true_class
    return pockets_predicted, pockets_true


@njit(cache=True)
def calib_range(y_true, y_pred, nbins):
    n = len(y_true)
    pockets_predicted = np.zeros(nbins, dtype=np.int64)
    pockets_true = np.zeros(nbins, dtype=np.int64)
    min_val, max_val = 1.0, 0.0
    for k in range(n):
        v = y_pred[k]
        if v > max_val:
            max_val = v
        if v < min_val:
            min_val = v
    span = max_val - min_val
    if span > 0:
        multiplier = (nbins - 1) / span
        for k in range(n):
            v = y_pred[k]
            ind = floor((v - min_val) * multiplier)
            pockets_predicted[ind] += 1
            pockets_true[ind] += y_true[k]
    return pockets_predicted, pockets_true


@njit(cache=True)
def prec_zip(y_true, y_pred, nclasses):
    allpreds = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)
    for true_class, predicted_class in zip(y_true, y_pred):
        if 0 <= predicted_class < nclasses:
            allpreds[predicted_class] += 1
            if predicted_class == true_class:
                hits[predicted_class] += 1
    return hits / np.maximum(allpreds, 1)


@njit(cache=True)
def prec_range(y_true, y_pred, nclasses):
    n = len(y_true)
    allpreds = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)
    for k in range(n):
        pc = y_pred[k]
        if 0 <= pc < nclasses:
            allpreds[pc] += 1
            if pc == y_true[k]:
                hits[pc] += 1
    return hits / np.maximum(allpreds, 1)


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
    print("== fast_calibration_binning ==")
    for n in [50_000, 200_000, 1_000_000]:
        yt = rng.integers(0, 2, size=n).astype(np.int64)
        yp = rng.random(n).astype(np.float64)
        t_z, _ = bench("zip",   calib_zip,   (yt, yp, 100))
        t_r, _ = bench("range", calib_range, (yt, yp, 100))
        a, b = calib_zip(yt, yp, 100); c, d = calib_range(yt, yp, 100)
        eq = np.array_equal(a, c) and np.array_equal(b, d)
        print(f"  n={n:>7}: zip={t_z:7.1f}us  range={t_r:7.1f}us  ({t_z/t_r:.2f}x)  eq={eq}")

    print()
    print("== fast_precision ==")
    for n in [50_000, 200_000, 1_000_000]:
        yt = rng.integers(0, 3, size=n).astype(np.int64)
        yp = rng.integers(0, 3, size=n).astype(np.int64)
        t_z, _ = bench("zip",   prec_zip,   (yt, yp, 3))
        t_r, _ = bench("range", prec_range, (yt, yp, 3))
        a = prec_zip(yt, yp, 3); b = prec_range(yt, yp, 3)
        eq = np.allclose(a, b)
        print(f"  n={n:>7}: zip={t_z:7.1f}us  range={t_r:7.1f}us  ({t_z/t_r:.2f}x)  eq={eq}")
