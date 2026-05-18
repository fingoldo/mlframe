"""T3#22 + T3#23 2026-05-18: Hermite Optuna RAM + MRMR DataFrame mutation memory.

Two related concerns flagged during the audit-fixes wave:

T3#22 (Hermite RAM)
-------------------
``optimise_hermite_pair`` evaluates the polynomial basis for every
trial. On a 4M-row x_a / x_b, each basis evaluation allocates an
``(n, degree+1)`` array (Hermite functions up to ``max_degree=4`` is
5 columns -> 160 MB per call at float64). Trial-by-trial allocation
without reuse can push RSS past 2 GB even with modest n_trials. This
benchmark measures peak RAM during a representative call so the
operator knows the cost.

T3#23 (DataFrame mutation memory)
---------------------------------
After Hermite/Chebyshev injection MRMR mutates the input X in place:

    X[_new_col_name] = _t_vals       # pandas
    X.with_columns(pl.Series(...))   # polars (returns a NEW frame)

For a wide pandas frame the per-column assignment is in-place. For
polars the ``with_columns`` returns a new frame so the OLD one is
either GC'd promptly (RSS no-op) or held by the caller (RSS doubles).
This benchmark measures the delta for both paths to inform users
choosing the input frame type.

Findings 2026-05-18 (printed to stdout, not asserted):
- Hermite n_trials=20 on n=400k: peak RAM delta ~XXX MB
- pandas in-place column add on n=400k * 50 cols: ~XXX MB
- polars ``with_columns`` on same: ~XXX MB (held vs released)

Run: python profiling/bench_hermite_ram_and_df_mutation.py
"""
from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd


def _rss_mb() -> float:
    """Resident set size of the current process in MB. Cross-platform via
    psutil when available, else /proc/self/status fallback for Linux."""
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 ** 2)
    except ImportError:
        if sys.platform.startswith("linux"):
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        return float("nan")


def bench_hermite_ram(n: int = 400_000, n_trials: int = 20) -> None:
    """T3#22: measure peak RSS during a representative ``optimise_hermite_pair`` call."""
    from mlframe.feature_selection.filters.hermite_fe import (
        optimise_hermite_pair,
    )

    print(f"[T3#22] Hermite RAM bench (n={n:_}, n_trials={n_trials})")
    print(f"  RSS before generate: {_rss_mb():.1f} MB")

    rng = np.random.default_rng(42)
    x_a = rng.normal(size=n).astype(np.float64)
    x_b = rng.normal(size=n).astype(np.float64)
    y = (np.sign(x_a * x_b) > 0).astype(np.int64)

    print(f"  RSS after data alloc: {_rss_mb():.1f} MB")
    rss_pre = _rss_mb()

    result = optimise_hermite_pair(
        x_a=x_a, x_b=x_b, y=y,
        basis="hermite",
        max_degree=4,
        n_trials=n_trials,
        optimizer="cma",
        warm_start=True,
        use_trivial_baseline=False,
        baseline_uplift_threshold=0.0,
        seed=42,
    )
    rss_post = _rss_mb()
    delta = rss_post - rss_pre
    print(f"  RSS after optimise:  {rss_post:.1f} MB  (delta +{delta:+.1f} MB)")
    if result is not None:
        print(f"  best MI = {result.mi:.4f}, degree_a = {result.degree_a}")
    else:
        print(f"  optimiser returned None")

    del x_a, x_b, y, result
    gc.collect()
    print(f"  RSS after gc: {_rss_mb():.1f} MB")


def bench_dataframe_mutation(n: int = 400_000, n_engineered: int = 50) -> None:
    """T3#23: measure delta RSS when adding ``n_engineered`` columns to a
    pandas frame in-place vs to a polars frame via ``with_columns``."""
    import polars as pl

    rng = np.random.default_rng(42)
    raw = {f"f{i}": rng.normal(size=n).astype(np.float32) for i in range(20)}
    print(f"\n[T3#23] DataFrame mutation memory bench "
          f"(n={n:_}, n_engineered={n_engineered})")

    # ---- pandas in-place ----
    df_pd = pd.DataFrame(raw)
    gc.collect()
    rss_pre = _rss_mb()
    for i in range(n_engineered):
        df_pd[f"engineered_{i}"] = rng.normal(size=n).astype(np.float64)
    rss_post = _rss_mb()
    print(f"  pandas in-place add: RSS {rss_pre:.1f} -> {rss_post:.1f} MB "
          f"(delta +{rss_post - rss_pre:.1f} MB; expected ~"
          f"{n * n_engineered * 8 / (1024 ** 2):.0f} MB)")
    del df_pd
    gc.collect()

    # ---- polars with_columns (returns new frame) ----
    df_pl = pl.from_pandas(pd.DataFrame(raw))
    gc.collect()
    rss_pre = _rss_mb()
    for i in range(n_engineered):
        new_col = pl.Series(f"engineered_{i}", rng.normal(size=n))
        df_pl = df_pl.with_columns(new_col)
    rss_post = _rss_mb()
    print(f"  polars with_columns: RSS {rss_pre:.1f} -> {rss_post:.1f} MB "
          f"(delta +{rss_post - rss_pre:.1f} MB)")


def main() -> int:
    print("=" * 70)
    print("T3#22 + T3#23 Hermite RAM & DataFrame mutation memory benchmarks")
    print("=" * 70)
    bench_hermite_ram(n=400_000, n_trials=20)
    bench_dataframe_mutation(n=400_000, n_engineered=50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
