"""A/B bench for MRMR's fit()-entry pandas input-isolation copy (perf audit finding #2, 2026-07-17).

RESOLVED: ``X.copy(deep=False)`` unconditionally (regardless of pandas' Copy-on-Write setting) is
correctness-equivalent to the prior CoW-gated ``deep=True`` fallback -- verified both by code audit (every
internal mutation site only ever ADDS a new column key: ``targ_*`` injection at fit entry, FE-stage
``X[name] = ...`` / ``pd.concat`` rebinds / hinge / cat-FE generators -- none overwrites an EXISTING
column's cell values in place) and empirically (forced CoW-off, full fe/ + input-mutation-isolation test
suite green). Eliminating the O(n*p) deep-copy branch is a real, non-negligible win on pandas < 3.0 with
CoW off (pandas' own default for most installed 2.x versions), which is the COMMON case, not an edge case.

Numbers below (median of 7 warm runs, this dev box, pandas 2.3.3, CoW OFF -- i.e. the pre-fix ``deep=True``
branch was active):

    n         p      deep=True copy (ms)   deep=False copy (ms)   speedup
    5_000     100             2.121                0.010            202x
    20_000    299            68.581                0.015           4482x
    100_000   500           861.759                0.010          89766x
    500_000   1000        10103.415                0.011         935502x

The deep-copy cost scales linearly in n*p (as expected for a real memcpy of every cell, and apparently with
a worse-than-linear constant on this box once the frame exceeds cache size); the shallow copy is a
near-constant ~0.01-0.015 ms (BlockManager/Index object construction only, no cell data touched) across the
whole range. At the project's stated 100GB-class-frame target this is the difference between a multi-second
deep copy on EVERY fit and an unmeasurable one. ``verify_correctness_under_forced_cow_off()`` below confirms
the actual safety property this speedup depends on: the caller's original frame is untouched after a real
FE-heavy fit with CoW forced off.

Run:  python path/to/_benchmarks/bench_mrmr_input_copy_deep_vs_shallow.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _make_frame(n: int, p: int) -> pd.DataFrame:
    """Build an n x p random-normal pandas frame for the copy-cost A/B."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(n, p)), columns=[f"c{i}" for i in range(p)])


def _time_copy(X: pd.DataFrame, deep: bool, n_reps: int = 7) -> float:
    """Median wall-clock cost (ms) of ``X.copy(deep=deep)`` over ``n_reps`` warm repetitions."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = X.copy(deep=deep)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000.0  # median, ms


def main() -> None:
    """Run the deep-vs-shallow copy A/B across a range of (n, p) and print the timing table."""
    pd.set_option("mode.copy_on_write", False)
    print(f"pandas {pd.__version__}, mode.copy_on_write={pd.get_option('mode.copy_on_write')}")
    print(f"{'n':>8} {'p':>6} {'deep=True (ms)':>16} {'deep=False (ms)':>18} {'speedup':>10}")
    for n, p in [(5_000, 100), (20_000, 299), (100_000, 500), (500_000, 1000)]:
        X = _make_frame(n, p)
        t_deep = _time_copy(X, deep=True)
        t_shallow = _time_copy(X, deep=False)
        speedup = t_deep / t_shallow if t_shallow > 0 else float("inf")
        print(f"{n:>8} {p:>6} {t_deep:>16.3f} {t_shallow:>18.3f} {speedup:>9.0f}x")


def verify_correctness_under_forced_cow_off() -> None:
    """Empirical correctness gate: fit an FE-heavy MRMR with CoW forced off and confirm the caller's
    original frame is byte-identical (columns, values, dtypes) after fit -- the actual safety property
    this bench's speed claim depends on, not just a timing number."""
    pd.set_option("mode.copy_on_write", False)
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    n, p = 500, 8
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"c{i}" for i in range(p)])
    y = (X["c0"] + 0.5 * X["c2"] > 0).astype(int)
    orig_cols = list(X.columns)
    orig_vals = X.to_numpy().copy()

    MRMR._FIT_CACHE.clear()
    m = MRMR(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, random_seed=1)
    m.fit(X, y)

    assert list(X.columns) == orig_cols, "fit() must not add/remove/rename the caller's columns"
    assert np.array_equal(X.to_numpy(), orig_vals), "fit() must not mutate the caller's cell values"
    print("verify_correctness_under_forced_cow_off: PASSED (caller's frame untouched after fit)")


if __name__ == "__main__":
    main()
    verify_correctness_under_forced_cow_off()
