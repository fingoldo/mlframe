"""Wall + cProfile A/B of MRMR ``fe_fast_search=True`` vs the exhaustive default on the two canonical interaction
synthetics used by ``tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_fast_search.py``.

Usage::

    python -m mlframe.feature_selection.filters._benchmarks.bench_fe_fast_search_vs_exhaustive --case 1 --mode fast --repeats 3
    python -m mlframe.feature_selection.filters._benchmarks.bench_fe_fast_search_vs_exhaustive --case 1 --mode ref --profile out.prof

Run each mode in its own process (numba/JIT warm state and MRMR's content-hash fit memo would otherwise leak
between modes). A warm-up fit on a different seed runs first so JIT compilation is not billed to the timed fits.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import time

import numpy as np
import pandas as pd


def make_case(case: int, n: int = 20_000, seed: int = 0):
    """Same generator as the biz-value test."""
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    if case == 1:
        y = a**2 / b + f / 5 + np.log(c) * np.sin(d)
    else:
        y = 0.2 * a**2 / b + f / 5 + np.log(c * 2) * np.sin(d / 3)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), pd.Series(y, name="y")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--mode", choices=["fast", "ref", "steps2"], default="fast")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    from mlframe.feature_selection.filters import MRMR

    # ref = the package default (fe_fast_search=False); steps2 = the genuinely exhaustive search (the step-2
    # fusion pass fe_fast_search used to skip, which became the default's behaviour when fe_max_steps went 2 -> 1).
    kw = dict(verbose=0, random_seed=0, fe_fast_search=args.mode == "fast")
    if args.mode == "steps2":
        kw["fe_max_steps"] = 2
    wdf, wy = make_case(args.case, n=4000, seed=123)
    MRMR(**kw).fit(wdf, wy)

    df, y = make_case(args.case, n=args.n)
    walls, cpus = [], []
    for r in range(args.repeats):
        prof = cProfile.Profile() if (args.profile and r == 0) else None
        t0 = time.perf_counter()
        c0 = time.process_time()
        if prof:
            prof.enable()
        MRMR._FIT_CACHE.clear()  # otherwise repeats replay the memoised fit
        m = MRMR(**kw).fit(df.copy(), y.copy())
        if prof:
            prof.disable()
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
        if prof:
            prof.dump_stats(args.profile)
            pstats.Stats(prof).sort_stats("cumulative").print_stats(40)
        print(f"repeat {r}: wall={walls[-1]:.2f}s cpu={cpus[-1]:.2f}s", flush=True)
    print(f"case={args.case} mode={args.mode} n={args.n} walls={[round(w, 2) for w in walls]} median={statistics.median(walls):.2f}s cpu={[round(c, 2) for c in cpus]} cpu_median={statistics.median(cpus):.2f}s")
    print("selected:", list(m.get_feature_names_out()))


if __name__ == "__main__":
    main()
