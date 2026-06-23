"""Bench for the dedup-aware object/string column cleaning map (cleaning_helpers.map_elementwise_dedup).

OLD = ``Series.map(callable)`` (one fcn call per row). NEW = map over unique values + reindex, gated by a
uniform-stride cardinality probe so the all-distinct case never regresses.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_obj_clean_dedup``

Measured (n=2M, 3.14.3, best-of-3 wall):
  low-200              OLD ~560ms  NEW ~265ms  2.11x   identical
  mid-5000             OLD ~545ms  NEW ~317ms  1.72x   identical
  with-none            OLD ~455ms  NEW ~228ms  1.99x   identical
  all-unique           OLD ~560ms  NEW ~575ms  0.98x   identical (gated to plain map)
  head-dup-tail-unique OLD ~550ms  NEW ~550ms  1.00x   identical (stride probe defeats head clustering)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.preprocessing.cleaning_helpers import map_elementwise_dedup


def _clean_fcn(s):
    return s.strip().lower() if isinstance(s, str) else s


def _best(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t)
    return min(ts), r


def main():
    rng = np.random.default_rng(0)
    n = 2_000_000
    cases = {
        "low-200": pd.Series(rng.choice([f" V{i} " for i in range(200)], size=n), dtype=object),
        "mid-5000": pd.Series(rng.choice([f" S{i} " for i in range(5000)], size=n), dtype=object),
        "with-none": pd.Series(rng.choice([" A ", " B ", None, " C "], size=n), dtype=object),
        "all-unique": pd.Series([f" V{i} " for i in range(n)], dtype=object),
        "head-dup-tail-unique": pd.Series([" DUP "] * (n // 2) + [f" T{i} " for i in range(n // 2)], dtype=object),
    }
    for label, col in cases.items():
        t_old, r_old = _best(lambda c=col: c.map(_clean_fcn))
        t_new, r_new = _best(lambda c=col: map_elementwise_dedup(c, _clean_fcn))
        print(f"{label:22s} OLD={t_old*1000:7.1f}ms NEW={t_new*1000:7.1f}ms speedup={t_old/t_new:.2f}x identical={r_old.equals(r_new)}")


if __name__ == "__main__":
    main()
