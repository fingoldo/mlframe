"""MEDIUM#12 2026-05-18: cProfile new audit-fix features.

Per the project memory rule "every new feature ships with unit tests +
biz_value quantitative-win test + cProfile hotspot optimization", profile
the three audit-fix features I shipped that touch hot paths:

1. ``_apply_hermite_pair`` - replay called from MRMR.transform on every
   test row; could be hot on n=4M.
2. ``recover_composite_y_scale_metrics`` - one-off helper; not hot.
3. ``_run_composite_target_wrapping`` watchdog block - per (entry, split)
   inside the wrap pass; already measured in
   ``bench_pack_g_watchdog_overhead.py``.

This script profiles #1 (the only candidate for material hot-path cost
where I haven't already measured) at n=400_000.

Findings (printed to stdout):
- Total wall time + top-20 by cumulative time
- Any function that takes >10% of total = hotspot worth optimising

Run: python profiling/bench_audit_fixes_cprofile.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys

import numpy as np
import pandas as pd


def _build_recipe_and_data(n: int = 400_000):
    from mlframe.feature_selection.filters.engineered_recipes import (
        build_hermite_pair_recipe,
    )
    from mlframe.feature_selection.filters.hermite_fe import HermiteResult

    rng = np.random.default_rng(0)
    x_a = rng.normal(size=n).astype(np.float64)
    x_b = rng.normal(size=n).astype(np.float64)
    result = HermiteResult(
        coef_a=np.array([0.0, 1.0, 0.5, -0.2], dtype=np.float64),
        coef_b=np.array([0.0, 1.0, 0.3, 0.0], dtype=np.float64),
        bin_func_name="mul",
        bin_func=np.multiply,
        mi=0.5, baseline_mi=0.3, uplift=1.67,
        degree_a=3, degree_b=3,
        basis="hermite",
        preprocess_a={"mean": float(x_a.mean()), "std": float(x_a.std() or 1.0)},
        preprocess_b={"mean": float(x_b.mean()), "std": float(x_b.std() or 1.0)},
    )
    recipe = build_hermite_pair_recipe(
        name="hermite_perf_bench",
        src_names=("x_a", "x_b"),
        hermite_result=result,
    )
    df = pd.DataFrame({"x_a": x_a, "x_b": x_b})
    return recipe, df


def main() -> int:
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    print("=" * 70)
    print("MEDIUM#12 cProfile - audit-fix new features hot-path check")
    print("=" * 70)
    print()

    recipe, df = _build_recipe_and_data(n=400_000)

    # Warm up (basis lookups, numba JIT).
    _ = apply_recipe(recipe, df)

    prof = cProfile.Profile()
    prof.enable()
    for _ in range(20):
        _ = apply_recipe(recipe, df)
    prof.disable()

    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(20)
    print(f"[apply_recipe(hermite_pair) x 20 calls on n=400_000]")
    print(stream.getvalue())

    return 0


if __name__ == "__main__":
    sys.exit(main())
