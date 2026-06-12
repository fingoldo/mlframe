"""Microbench: per-class ICE reuse in the probabilistic report.

The multiclass report called ``compute_probabilistic_multiclass_error`` once on
the full (N, K) probs, then AGAIN per-class on each 1-D column inside a K-loop.
The batched kernel already computes the per-class ICE vector during the full
call and discarded it. ``return_per_class=True`` exposes that (K,) vector so the
report indexes it instead of recomputing -- bit-identical (same kernel inputs
per class), saving K redundant kernel dispatches.

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.metrics._benchmarks.bench_per_class_ice_reuse

Measured (warm, min-of-5, this dev box) -- maxdiff old-vs-new per-class = 0.0 (bit-identical):
    N=10000   K=8  before=18.4ms    after=2.8ms    6.70x
    N=200000  K=8  before=475.0ms   after=97.2ms   4.89x
    N=1000000 K=5  before=1436.8ms  after=344.6ms  4.17x
"""
from __future__ import annotations

import timeit

import numpy as np

from mlframe.metrics.core import compute_probabilistic_multiclass_error as ice


def _data(n, k, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k))
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return y_true, p


def _old(y_true, p, k):
    ice(y_true=y_true, y_score=p, nbins=10)  # full aggregate call
    return [ice(y_true=(y_true == c).astype(np.int8), y_score=p[:, c], nbins=10) for c in range(k)]


def _new(y_true, p, k):
    _, per = ice(y_true=y_true, y_score=p, nbins=10, return_per_class=True)
    return [per[c] for c in range(k)]


def main():
    for n, k in [(10_000, 8), (200_000, 8), (1_000_000, 5)]:
        y_true, p = _data(n, k)
        _old(y_true, p, k)
        _new(y_true, p, k)  # warm
        maxdiff = max(abs(a - b) for a, b in zip(_old(y_true, p, k), _new(y_true, p, k)))
        to = min(timeit.repeat(lambda: _old(y_true, p, k), number=3, repeat=5)) / 3 * 1000
        tn = min(timeit.repeat(lambda: _new(y_true, p, k), number=3, repeat=5)) / 3 * 1000
        print(f"N={n} K={k}  before={to:.1f}ms  after={tn:.1f}ms  {to / tn:.2f}x  maxdiff={maxdiff}")


if __name__ == "__main__":
    main()
