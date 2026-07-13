"""Paired interleaved A/B for is_variable_truly_continuous @200k.
NEW = current cleaning._get_nunique. OLD = the exact np.unique impl, monkeypatched in.
Alternates OLD/NEW back-to-back across N trials; reports paired wins + min/median so shared-machine noise cancels.
"""
import time
import scipy.stats  # noqa: F401
import numba  # noqa: F401
import numpy as np
import mlframe.preprocessing.cleaning as C
from mlframe.preprocessing.cleaning import is_variable_truly_continuous

NEW = C._get_nunique


def OLD(vals, skip_nan=True, skip_vals=None):
    unique_vals = np.unique(vals)
    if skip_nan:
        if unique_vals.dtype.kind in ("f", "c"):
            unique_vals = unique_vals[~np.isnan(unique_vals)]
        else:
            import pandas as pd
            unique_vals = unique_vals[~pd.isna(unique_vals)]
    if skip_vals:
        for val in skip_vals:
            unique_vals = unique_vals[unique_vals != val]
    return len(unique_vals)


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


    def run_once():
        out = []
        for v in cols.values():
            out.append(is_variable_truly_continuous(values=v, use_quantile=0.1))
        return out


    # identity
    C._get_nunique = OLD
    r_old = run_once()
    C._get_nunique = NEW
    r_new = run_once()
    print("IDENTICAL", r_old == r_new)

    # warm njit
    run_once()

    REP = 50
    told, tnew, wins = [], [], 0
    for _ in range(REP):
        C._get_nunique = OLD
        t0 = time.perf_counter(); run_once(); a = time.perf_counter() - t0
        C._get_nunique = NEW
        t0 = time.perf_counter(); run_once(); b = time.perf_counter() - t0
        told.append(a); tnew.append(b)
        if b < a:
            wins += 1
    told.sort(); tnew.sort()
    print(f"NEW faster in {wins}/{REP} paired trials")
    print(f"OLD min={told[0]*1e3:.2f}ms median={told[len(told)//2]*1e3:.2f}ms")
    print(f"NEW min={tnew[0]*1e3:.2f}ms median={tnew[len(tnew)//2]*1e3:.2f}ms")
    print(f"speedup median={told[len(told)//2]/tnew[len(tnew)//2]:.3f}x  min={told[0]/tnew[0]:.3f}x")
