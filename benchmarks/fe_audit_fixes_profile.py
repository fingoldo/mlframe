"""cProfile baselines for the 2026-05-14 feature_engineering audit fixes.

Per project convention every new feature ships with unit + biz_value + cProfile attribution.
Run via ``python -m mlframe.benchmarks.fe_audit_fixes_profile`` from the project root; outputs
go to D:/Temp/fe_audit_profile_<feature>.prof and a JSON summary alongside.

Covers:
- ``rolling_moving_average`` (numerical.py) - Kahan-compensated rolling mean (fastmath OFF).
- ``_kfold_target_encode`` (bruteforce.py) - new leakage-free OOF encoder helper.
- ``find_best_mps_sequence`` (mps.py) - DP after the _trade_count fix.
- ``welford_moments_seq`` (_numerical_stable.py) - Pebay 2008 single-pass moments.
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

_OUT_DIR = Path("D:/Temp")
_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _profile_one(name: str, fn: Callable[[], None], reps: int = 3) -> Dict[str, float]:
    """Run ``fn`` once for warm-up, then ``reps`` more times under cProfile; persist .prof and return
    the top-10 cumulative-time rows for JSON summary."""
    fn()  # warm-up (numba JIT happens here, NOT in the profile)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(reps):
        fn()
    profiler.disable()

    prof_path = _OUT_DIR / f"fe_audit_profile_{name}.prof"
    profiler.dump_stats(str(prof_path))

    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats("cumulative").print_stats(10)
    summary = buf.getvalue()
    print(f"\n=== {name} (reps={reps}) ===")
    print(summary)

    # Extract top function by cumulative time for the JSON ledger.
    stats = pstats.Stats(profiler)
    total = sum(rec[3] for rec in stats.stats.values())  # cumtime sum (proxy for wall)
    return {
        "name": name,
        "reps": reps,
        "total_cumtime_s": total,
        "summary_first_lines": summary.splitlines()[:20],
        "prof_path": str(prof_path),
    }


def bench_rolling_moving_average() -> Dict[str, float]:
    from mlframe.feature_engineering.numerical import rolling_moving_average
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(100_000).astype(np.float64)

    def run():
        rolling_moving_average(arr, n=200)

    return _profile_one("rolling_moving_average", run, reps=10)


def bench_welford_moments() -> Dict[str, float]:
    from mlframe.feature_engineering._numerical_stable import welford_moments_seq
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(100_000).astype(np.float64)

    def run():
        welford_moments_seq(arr)

    return _profile_one("welford_moments_seq", run, reps=10)


def bench_find_best_mps_sequence() -> Dict[str, float]:
    from mlframe.feature_engineering.mps import find_best_mps_sequence
    rng = np.random.default_rng(0)
    prices = 100.0 + np.cumsum(rng.standard_normal(5_000)).astype(np.float64)

    def run():
        find_best_mps_sequence(
            prices=prices,
            raw_prices=prices,
            tc=3e-4,
            tc_mode_is_fraction=True,
            optimize_consecutive_regions=True,
        )

    return _profile_one("find_best_mps_sequence", run, reps=5)


def bench_kfold_target_encode() -> Dict[str, float]:
    try:
        from category_encoders import CatBoostEncoder  # noqa: F401
    except ImportError:
        print("category_encoders not installed - skipping kfold_target_encode benchmark")
        return {"name": "_kfold_target_encode", "skipped": True}
    from mlframe.feature_engineering.bruteforce import _kfold_target_encode

    rng = np.random.default_rng(0)
    n = 50_000
    df = pd.DataFrame({"cat": rng.choice(list("abcdefghij"), size=n)})
    df["cat"] = df["cat"].astype("category")
    target = pd.Series(rng.standard_normal(n))

    def run():
        _kfold_target_encode(df, cols=["cat"], target=target, n_splits=5, random_state=0)

    return _profile_one("_kfold_target_encode", run, reps=3)


def main() -> int:
    benches = [
        bench_rolling_moving_average,
        bench_welford_moments,
        bench_find_best_mps_sequence,
        bench_kfold_target_encode,
    ]
    results = [b() for b in benches]

    summary_path = _OUT_DIR / "fe_audit_profile_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote profile summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
