"""Fit a default MRMR on 20k x 40 and list every numba kernel that really compiled (cache miss or uncacheable), slowest first.

Run twice: the second (warm) run must report no misses; any name printed is a kernel recompiled on every process start.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def main() -> None:
    """Fit, recording every real numba compile (a cache miss or an uncacheable kernel) with its qualified name and compile seconds."""
    warnings.simplefilter("ignore")
    from numba.core import event

    rec = event.RecordingListener()
    event.register("numba:compile", rec)
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20000, 40)), columns=[f"f{i}" for i in range(40)])
    y = (X["f0"] * X["f1"] + np.sin(X["f2"]) + 0.3 * rng.normal(size=20000) > 0).astype(int)
    MRMR(verbose=0, n_jobs=1).fit(X, y)
    starts: dict = {}
    rows = []
    for ts, ev in rec.buffer:
        key = id(ev.data["dispatcher"]), str(ev.data["args"])
        if ev.is_start:
            starts[key] = ts
        elif key in starts:
            d = ev.data["dispatcher"]
            rows.append((ts - starts.pop(key), f"{d.py_func.__module__}.{d.py_func.__qualname__}"))
    rows.sort(reverse=True)
    print("COMPILED", len(rows), "total_s", round(sum(r[0] for r in rows), 1))
    for sec, name in rows:
        print(f"{sec:8.2f}s {name}")


if __name__ == "__main__":
    main()
