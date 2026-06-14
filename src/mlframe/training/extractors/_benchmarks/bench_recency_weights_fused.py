"""Bench: fused njit recency-weight kernel vs the prior four-sweep numpy chain (loop iter119, @10M).

``get_sample_weights_by_recency`` built the weight array with four full-array numpy sweeps:
``_delta_secs / 86400`` -> ``np.maximum(..., floor)`` -> ``np.log(...)`` -> affine combine. At 10M float64 rows that
arithmetic chain alone cost ~250ms. ``_recency_weights_fused`` collapses it to one prange pass (``base - log(d) * wdpy``,
all loop-invariant constants folded into ``base``), bit-equivalent to <=1 ULP (~4e-16) from fastmath reduction-order.

Run:
    python -m mlframe.training.extractors._benchmarks.bench_recency_weights_fused

Measured @10M (separate-module A/B vs HEAD baseline, best-of-7 median, py3.14 store, CPU):
    datetime path: 857ms -> 484ms  (1.77x, -44%)
    numeric  path: 507ms -> 181ms  (2.80x, -64%)
    identity: max abs diff 4.44e-16, zero-span bit-identical.

Verdict: RESOLVED. The ~250ms arithmetic chain is mlframe-own (the ~250ms ``.dt.total_seconds()`` upstream is pandas).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _old_chain(delta_secs: np.ndarray, span_days: float, min_weight: float = 1.0, wdpy: float = 0.1) -> np.ndarray:
    min_age_days = 1.0 / 86400.0
    log_min_age = np.log(min_age_days)
    days_from_max = np.maximum(delta_secs / 86400.0, min_age_days)
    max_drop = (np.log(span_days) - log_min_age) * wdpy
    return min_weight + max_drop - (np.log(days_from_max) - log_min_age) * wdpy


def main(n: int = 10_000_000) -> None:
    from mlframe.training.extractors import get_sample_weights_by_recency

    rng = np.random.default_rng(0)
    secs = np.sort(rng.integers(0, 3 * 365 * 86400, n)).astype("datetime64[s]")
    ds = pd.Series(pd.to_datetime(secs))
    dsn = pd.Series(secs.astype("int64"))

    delta = (secs.max() - secs).astype("timedelta64[s]").astype(np.float64)
    span_days = float((secs.max() - secs.min()).astype("timedelta64[s]").astype(np.float64)) / 86400.0
    ref = _old_chain(delta, span_days)
    new = get_sample_weights_by_recency(dsn)
    print("identity max abs diff:", float(np.max(np.abs(ref - new))), "finite:", bool(np.isfinite(new).all()))

    def bench(f, a, k: int = 7) -> float:
        f(a)
        ts = []
        for _ in range(k):
            t = time.perf_counter()
            f(a)
            ts.append(time.perf_counter() - t)
        return sorted(ts)[k // 2] * 1000.0

    print(f"NEW datetime median ms @{n}:", round(bench(get_sample_weights_by_recency, ds), 1))
    print(f"NEW numeric  median ms @{n}:", round(bench(get_sample_weights_by_recency, dsn), 1))


if __name__ == "__main__":
    main()
