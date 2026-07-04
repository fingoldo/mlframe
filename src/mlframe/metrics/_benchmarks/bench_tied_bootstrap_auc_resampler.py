"""Bench: tied-score bootstrap AUC resampler fast path (grouped) vs exact argsort fallback.

The tied-score fallback of ``make_bootstrap_auc_resampler`` used to re-argsort the whole
n-length resampled vector every bootstrap resample (O(n log n) x ~3000). The grouped path
(``_fused_resample_auc_grouped``) scores each resample in O(n + K) over the K distinct-score
groups, bit-identical to ``fast_roc_auc_unstable(y[idx], score[idx])`` because AUC is invariant
to the within-tie argsort order.

Run: python -m mlframe.metrics._benchmarks.bench_tied_bootstrap_auc_resampler
"""
from __future__ import annotations

import time
import numpy as np

from mlframe.metrics._core_auc_brier import (
    make_bootstrap_auc_resampler,
    fast_roc_auc_unstable,
    _fused_resample_auc_grouped,
)


def _make_tied(n: int, ndistinct: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # low-cardinality binned classifier: probs snapped to ndistinct levels
    raw = rng.random(n)
    score = np.round(raw * (ndistinct - 1)) / (ndistinct - 1)
    # label correlated with score so AUC is a real number, not 0.5
    y = (rng.random(n) < (0.2 + 0.6 * score)).astype(np.int64)
    return y, score


def _exact_resampler(y_true, y_score):
    def _r(idx):
        return fast_roc_auc_unstable(y_true[idx], y_score[idx])
    return _r


def bench(n=200_000, ndistinct=50, nboot=1000, seed=0):
    y, score = _make_tied(n, ndistinct, seed)
    rng = np.random.default_rng(123)
    resamples = [rng.integers(0, n, n) for _ in range(nboot)]

    fast = make_bootstrap_auc_resampler(y, score)
    exact = _exact_resampler(y, score)

    # identity
    maxdiff = 0.0
    for idx in resamples[:50]:
        a = fast(idx)
        b = exact(idx)
        d = abs(a - b)
        if d > maxdiff:
            maxdiff = d

    # warm numba
    fast(resamples[0]); exact(resamples[0])

    t0 = time.perf_counter()
    for idx in resamples:
        fast(idx)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    for idx in resamples:
        exact(idx)
    t_exact = time.perf_counter() - t0

    print(f"n={n} ndistinct={ndistinct} nboot={nboot}: fast={t_fast*1e3:.1f}ms exact={t_exact*1e3:.1f}ms "
          f"speedup={t_exact/t_fast:.2f}x maxdiff={maxdiff:.3e}")
    return t_exact / t_fast, maxdiff


if __name__ == "__main__":
    for n in (200_000, 1_000_000):
        for nd in (20, 200):
            bench(n=n, ndistinct=nd, nboot=500)
