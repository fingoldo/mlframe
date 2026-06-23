"""Bench: probability_separation_score SEQ kernel — alloc-heavy vs fused scalar.

The current ``_probability_separation_score_seq`` (in
``mlframe/metrics/_log_loss_and_separation.py``) builds a boolean mask
``idx = y_true == class_label`` (length-n alloc), then ``idx.sum()``, then a
fancy-indexed copy ``y_prob[idx]`` (alloc), then ``np.mean`` (full pass) and
``np.std`` (which itself builds another temporary + two passes). For the
in-class subset of size m this is ~3 array allocations and ~4 passes.

The fused variant walks ``y_true``/``y_prob`` ONCE accumulating count + sum,
then ONCE more for the centred SSE — zero allocations. This is the seq path
that serves all n < _PARALLEL_MULTILABEL_THRESHOLD (50k), i.e. every typical
classification/regression report shape.

Identity: bit-equivalent modulo FP reduction order; np.mean/np.std use
pairwise summation while the fused form is a plain running sum, so a
single-ULP (~1e-15 relative) delta is expected and acceptable (the score is
a reporting diagnostic, not a selection decision).

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.metrics._benchmarks.bench_prob_separation_seq_fused
"""
from __future__ import annotations

import time
import numpy as np
import numba

from mlframe.metrics._numba_params import NUMBA_NJIT_PARAMS


@numba.njit(**NUMBA_NJIT_PARAMS)
def _seq_old(y_true, y_prob, class_label=1, std_weight=0.5):
    idx = y_true == class_label
    if idx.sum() == 0:
        return np.nan
    res = np.mean(y_prob[idx])
    if std_weight != 0.0:
        addend = np.std(y_prob[idx]) * std_weight
        if class_label == 1:
            res = res - addend
        else:
            res = res + addend
    return res


@numba.njit(**NUMBA_NJIT_PARAMS)
def _seq_new(y_true, y_prob, class_label=1, std_weight=0.5):
    n = y_true.shape[0]
    n_in = 0
    s = 0.0
    for i in range(n):
        if y_true[i] == class_label:
            n_in += 1
            s += y_prob[i]
    if n_in == 0:
        return np.nan
    mean = s / n_in
    if std_weight == 0.0:
        return mean
    sse = 0.0
    for i in range(n):
        if y_true[i] == class_label:
            d = y_prob[i] - mean
            sse += d * d
    std = np.sqrt(sse / n_in)
    addend = std * std_weight
    if class_label == 1:
        return mean - addend
    return mean + addend


def _best_of(fn, args, reps=200):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def main():
    rng = np.random.default_rng(0)
    print(f"{'n':>10} {'cl':>3} {'old(us)':>10} {'new(us)':>10} {'speedup':>8} {'max|d|':>12}")
    for n in (2_000, 10_000, 49_000):
        for class_label in (1, 0):
            yt = rng.integers(0, 2, size=n).astype(np.float64)
            yp = rng.random(n)
            # warm
            o = _seq_old(yt, yp, class_label, 0.5)
            v = _seq_new(yt, yp, class_label, 0.5)
            d = abs(o - v)
            to = _best_of(lambda: None and 0, ()) if False else _best_of(_seq_old, (yt, yp, class_label, 0.5))
            tn = _best_of(_seq_new, (yt, yp, class_label, 0.5))
            print(f"{n:>10} {class_label:>3} {to*1e6:>10.2f} {tn*1e6:>10.2f} {to/tn:>8.2f}x {d:>12.2e}")


if __name__ == "__main__":
    main()
