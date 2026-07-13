"""Microbench: _get_nunique (np.unique full sort + post-filter) vs a njit count-distinct
that sorts once and counts distinct values skipping nan + skip_vals, never materializing the unique array.

The caller (is_variable_truly_continuous) only uses the COUNT, never the unique values themselves,
so the unique-array materialization + the trailing boolean-mask passes (unique_vals != val) are pure waste.
"""
import time
import numpy as np
import numba


@numba.njit(cache=True)
def _count_distinct_float(sorted_vals, skip0, skip1):
    n = sorted_vals.shape[0]
    count = 0
    prev = np.nan
    have_prev = False
    for i in range(n):
        v = sorted_vals[i]
        if np.isnan(v):
            continue
        if v == skip0 or (skip1 == skip1 and v == skip1):
            continue
        if (not have_prev) or v != prev:
            count += 1
            prev = v
            have_prev = True
    return count


def get_nunique_old(vals, skip_nan=True, skip_vals=None):
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


def get_nunique_new(vals, skip_vals):
    sv = np.sort(vals, kind="quicksort")
    if skip_vals is None:
        skip0 = np.nan
        skip1 = np.nan
    elif len(skip_vals) == 1:
        skip0 = float(skip_vals[0])
        skip1 = np.nan
    else:
        skip0 = float(skip_vals[0])
        skip1 = float(skip_vals[1])
    return _count_distinct_float(sv, skip0, skip1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 200000
    arrs = {
        "frac": np.modf(np.round(rng.uniform(0, 1000, N), 3))[0],
        "intpart": np.modf(rng.uniform(-1e6, 1e6, N))[1],
    }
    skipsets = {"frac": (0.0, 1.0), "intpart": (0.0,)}

    # correctness
    for k, a in arrs.items():
        o = get_nunique_old(a, skip_vals=skipsets[k])
        n = get_nunique_new(a, skipsets[k])
        print(f"{k}: old={o} new={n} match={o==n}")

    # warm njit
    get_nunique_new(arrs["frac"], (0.0, 1.0))

    REP = 200
    for k, a in arrs.items():
        sk = skipsets[k]
        t0 = time.perf_counter()
        for _ in range(REP):
            get_nunique_old(a, skip_vals=sk)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(REP):
            get_nunique_new(a, sk)
        t_new = time.perf_counter() - t0
        print(f"{k}: old={t_old/REP*1e3:.3f}ms new={t_new/REP*1e3:.3f}ms speedup={t_old/t_new:.2f}x")
