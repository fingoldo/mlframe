"""cProfile harness for per-query NDCG bootstrap CI (charts/ltr.py::bootstrap_ndcg_ci).

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_ndcg_bootstrap_ci``

The bootstrap resamples QUERIES (the independent unit), fully vectorised: one
``rng.integers((n_boot, n_eff))`` gather of resampled query indices into the
precomputed per-query NDCG vector, ``mean(axis=1)``, then two percentiles. No
python bootstrap loop. ``n_eff`` is capped at ``_BOOTSTRAP_QUERY_CAP`` so the
gather is bounded at ``(B, cap)`` regardless of query count (a million-query LTR
subsamples per resample instead of allocating a (1000, 1e6) = 8 GB index array).
At B=1000 / cap=50k the gather is the wall; ``rng.integers`` over 5e7 cells is the
floor, and the percentiles are a single partition over the B means. This harness
confirms the vectorised form stays well under a second at production query counts.

Measured @ n_queries=1e6, B=1000: ~444 ms/call. The whole cost is the (B, n_eff)
random-index generation + the float64 gather (capped at 1000x50000 = 400 MB) +
mean(axis=1); all C-level, no python bootstrap loop. No actionable speedup: this
IS the vectorised resample floor, and the 50k cap keeps the gather memory bounded
regardless of query count while preserving the 1/sqrt(n_queries) CI narrowing up
to the cap (a per-query NDCG over >50k queries is already pinned to <~0.005 width).
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.ltr import bootstrap_ndcg_ci


def main(n_queries: int = 1_000_000):
    rng = np.random.default_rng(0)
    per_query = rng.beta(5, 2, size=n_queries)  # per-query NDCG vector (cached upstream by the panel)
    bootstrap_ndcg_ci(per_query, n_boot=10)  # warmup

    t0 = time.perf_counter()
    for _ in range(5):
        bootstrap_ndcg_ci(per_query, n_boot=1000)
    wall = (time.perf_counter() - t0) / 5.0
    print(f"bootstrap_ndcg_ci @ n_queries={n_queries}, B=1000: {wall*1000:.1f} ms/call (best-of-5 mean)")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        bootstrap_ndcg_ci(per_query, n_boot=1000)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
