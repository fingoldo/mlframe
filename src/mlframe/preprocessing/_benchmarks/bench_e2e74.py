"""End-to-end A/B for is_variable_truly_continuous @200k: OLD (git HEAD) vs NEW (working tree).
Run via: python bench_e2e74.py old|new  -> prints median wall over many reps + per-col verdicts for identity.
"""
import sys, time
import scipy.stats  # noqa: F401
import numba  # noqa: F401
import numpy as np
from mlframe.preprocessing.cleaning import is_variable_truly_continuous

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 200000
    cols = {
        "cont": rng.normal(0, 1, N).astype(np.float64),
        "cont2": (rng.normal(100, 30, N)).astype(np.float64),
        "frac": np.round(rng.uniform(0, 1000, N), 3),
        "intlike": rng.integers(0, 500, N).astype(np.float64),
        "wide": rng.uniform(-1e6, 1e6, N).astype(np.float64),
    }
    # verdicts
    verdicts = {k: is_variable_truly_continuous(values=v, use_quantile=0.1) for k, v in cols.items()}
    print("VERDICTS", {k: (bool(a), round(float(b), 12)) for k, (a, b) in verdicts.items()})

    REP = 40
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        for v in cols.values():
            is_variable_truly_continuous(values=v, use_quantile=0.1)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f"median={ts[len(ts)//2]*1e3:.2f}ms min={ts[0]*1e3:.2f}ms")
