"""Bench: _temporal_agg_fe._entity_key_series per-row str() vs vectorized astype(str).

The entity-key cast collapses one or more entity columns into a per-row string
key, later fed to pd.factorize for dense group codes. The prior implementation
used a Python-level ``.astype(object).map(lambda v: str(v))`` (one Python call
per row); ``.astype(str)`` is the vectorized equivalent and bit-identical for
int/float/object/categorical columns.

Measured best-of-7 at n=200k (this box):
    int    old=  57ms  new=  49ms  1.16x
    float  old= 193ms  new= 202ms  0.96x (neutral, within noise)
    str    old=  29ms  new=   2.4ms 12.0x  <- entity ids are usually strings

Run: python -m mlframe.feature_selection.filters._benchmarks.bench_temporal_entity_key_astype_str
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _best(fn, k=7):
    t = []
    for _ in range(k):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return min(t)


def main(n: int = 200_000) -> None:
    rng = np.random.default_rng(0)
    cases = [
        ("int", rng.integers(0, 5000, n)),
        ("float", rng.normal(size=n)),
        ("str", rng.choice(["aa", "bb", "cc", "dd"], n)),
    ]
    for desc, col in cases:
        s = pd.Series(col)
        old = _best(lambda: s.astype(object).map(lambda v: str(v)))
        new = _best(lambda: s.astype(str))
        identical = s.astype(object).map(lambda v: str(v)).equals(s.astype(str))
        print(
            f"{desc:6s} old={old * 1000:8.2f}ms new={new * 1000:8.2f}ms "
            f"speedup={old / new:.2f}x identical={identical}"
        )


if __name__ == "__main__":
    main()
