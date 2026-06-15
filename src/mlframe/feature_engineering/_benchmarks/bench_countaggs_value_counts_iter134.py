"""iter134 perf-loop bench: value_counts backend for compute_countaggs @10M rows.

Workload @10M: ``compute_countaggs`` (feature_engineering/categorical.py) aggregates a series by its
value-count distribution. The ONLY O(n) cost in the whole function is line 54 ``arr.value_counts(normalize=...)``;
everything after operates on the (small) unique-value distribution. So at 10M rows the function's wall is the
value_counts hashtable build.

Lead investigated: replace pandas ``value_counts`` with ``np.unique(return_counts=True)`` + descending argsort.

Measured (this box, py3.14, store numpy):
  int1k   (ncat=1000):    pandas  89.6 ms   np.unique+sort 123.5 ms  (pandas WINS — hashtable beats O(n log n) sort at low card)
  int100k (ncat=100000):  pandas 388.2 ms   np.unique+sort 190.8 ms  (np WINS ~2.0x — pandas hashtable degrades at high card)

So the np.unique path is faster ONLY at high cardinality (a detectable crossover). BUT:

IDENTITY — REJECT. The downstream code emits ``top_value`` (``values[:top_n]``) and ``btm_value``
(``values[-top_n:]``) directly as feature values, ranked by count. pandas value_counts and np.argsort break
COUNT TIES differently (pandas = hashtable insertion order; np.argsort = value order). At high cardinality the
bottom-count region is dominated by singletons (count==1): in the 10M / 2M-unique probe, 67,556 values tied at the
minimum count, and the emitted ``btm_value`` diverged outright (pandas 1309489 vs np 1694742). That is a
SELECTION-ALTERING divergence on a feature value (~not a 1e-9 FP delta) — exactly the high-card regime where the
np path would win is the regime where the tie-break diverges most. The gate predicate ("no ties at the count
extremes") is neither cheap nor commonly satisfiable (singletons are the norm on high-card columns). Not gateable
into a clean win.

PRODUCTION-SHAPE — additionally moot. The only callers (``_timeseries_emit.py``) invoke ``compute_countaggs`` on
per-WINDOW sub-series (small), never on a full 10M-row column, so the value_counts cost is already amortised over
many tiny series — there is no 10M single-column hot path here to optimise.

Verdict: REJECT (identity selection-altering + not the prod shape). Run this bench to re-confirm the crossover and
the tie divergence on different hardware / numpy build.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.modules.setdefault("cupy", None)  # avoid py3.14 cold-import segfault under contention


def _bench(fn, reps: int = 5) -> float:
    fn()  # warm
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000.0)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 10_000_000
    for ncat, label in [(1000, "int1k"), (100_000, "int100k")]:
        a = rng.integers(0, ncat, size=n)
        arr = pd.Series(a)

        def np_uc(_a=a, _n=n):
            v, c = np.unique(_a, return_counts=True)
            cn = c / _n
            order = np.argsort(c)[::-1]
            return v[order], cn[order]

        t_pd = _bench(lambda _arr=arr: _arr.value_counts(normalize=True))
        t_np = _bench(np_uc)
        print(f"{label:8s} pandas {t_pd:7.1f} ms   np.unique+sort {t_np:7.1f} ms")

    # identity probe: bottom-value diverges under count ties at high cardinality.
    a = rng.integers(0, 2_000_000, size=n)
    arr = pd.Series(a)
    vc = arr.value_counts(normalize=True)
    v, c = np.unique(a, return_counts=True)
    order = np.argsort(c)[::-1]
    pd_btm, np_btm = vc.index.values[-1], v[order][-1]
    print(f"btm_value pandas={pd_btm} np={np_btm} match={pd_btm == np_btm} n_bottom_tied={(c == c.min()).sum()}")


if __name__ == "__main__":
    main()
