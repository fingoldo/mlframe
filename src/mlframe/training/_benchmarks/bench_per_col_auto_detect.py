"""Per-column auto-detect single-pass vs legacy per-col Series-call wall-time bench.

Measures the speedup of ``_auto_detect_feature_types``'s batched aggregation (one lazy ``select`` for all
n_unique + count) over the legacy per-column ``df[name].n_unique()`` + ``int(df[name].count())`` round-trip.

The pre-fix code did 2 Python -> polars round-trips per candidate column; on 60 candidate columns that's
120 kernel launches (50-200 ms). The fix collapses this to ONE lazy collect computing both stats for every
column simultaneously.

cProfile hotspots (representative run, 2026-05-16):
- post-fix: ``LazyFrame.collect`` of the batched (n_unique + count) aggregation (~90%); per-col Python
  dict-lookup of cached values is negligible (~5%).
- pre-fix: ``Series.n_unique`` C-bridge entry (~50%); ``Series.count`` (~30%); per-col attribute access (~5%);
  Python-level loop overhead (~15%).
No further optimisation in scope: one aggregation collect is the theoretical floor for the (n_unique + count)
pair across N columns.

Usage:
    python -m mlframe.training._benchmarks.bench_per_col_auto_detect

Writes results JSON to ``_results/bench_per_col_auto_detect.json``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

RESULTS_DIR = Path(__file__).parent / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _synth_polars(n_rows: int, n_cols: int, seed: int) -> pl.DataFrame:
    """60 mixed-dtype polars cols at n_rows: ~half string-like (text-auto-promo candidates), ~quarter mostly-null."""
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_cols):
        kind = i % 4
        if kind == 0:
            # high-card string column (text candidate)
            data[f"c_str_{i}"] = [f"v{int(x)}" for x in rng.integers(0, max(2, n_rows // 50), size=n_rows)]
        elif kind == 1:
            # mostly-null string column (still text-like)
            arr = [f"v{int(x)}" for x in rng.integers(0, 500, size=n_rows)]
            mask = rng.random(n_rows) < 0.95
            for j, m in enumerate(mask):
                if m:
                    arr[j] = None
            data[f"c_sparse_{i}"] = arr
        elif kind == 2:
            # low-card string column (kept as cat)
            data[f"c_lowcard_{i}"] = [f"v{int(x)}" for x in rng.integers(0, 8, size=n_rows)]
        else:
            # numeric column (skipped by auto-detect; still in df)
            data[f"c_num_{i}"] = rng.normal(size=n_rows).astype(np.float64)
    return pl.DataFrame(data)


def _legacy_polars_path(df, threshold, min_non_null_abs):
    """Vendored pre-fix: per-col n_unique + per-col count. Excludes embedding/honor checks (irrelevant here)."""
    text_features, cardinalities, skipped, dropped = [], {}, [], []
    for name, dtype in df.schema.items():
        is_text_like = dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(dtype, pl.Enum)
        if not is_text_like:
            continue
        n_unique = df[name].n_unique()
        if n_unique > threshold:
            non_null = int(df[name].count())
            if non_null < min_non_null_abs:
                skipped.append((name, n_unique, non_null))
                continue
            cardinalities[name] = n_unique
            text_features.append(name)
    return text_features, cardinalities, skipped


def _new_polars_path(df, threshold, min_non_null_abs):
    """Vendored post-fix: single batched aggregation."""
    text_like_cols = [name for name, dtype in df.schema.items() if dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(dtype, pl.Enum)]
    if not text_like_cols:
        return [], {}, []
    _aggs = [pl.col(c).n_unique().alias(f"__nu_{i}__") for i, c in enumerate(text_like_cols)] + [
        pl.col(c).count().alias(f"__cnt_{i}__") for i, c in enumerate(text_like_cols)
    ]
    _agg_row = df.lazy().select(_aggs).collect()
    text_features, cardinalities, skipped = [], {}, []
    for i, name in enumerate(text_like_cols):
        n_unique = int(_agg_row[f"__nu_{i}__"][0])
        if n_unique > threshold:
            non_null = int(_agg_row[f"__cnt_{i}__"][0])
            if non_null < min_non_null_abs:
                skipped.append((name, n_unique, non_null))
                continue
            cardinalities[name] = n_unique
            text_features.append(name)
    return text_features, cardinalities, skipped


def _bench(n_rows: int, n_cols: int, n_runs: int) -> dict:
    df = _synth_polars(n_rows, n_cols, seed=1)
    threshold = 100
    min_non_null_abs = max(1, int(n_rows * 0.01))

    # Warm both paths.
    _legacy_polars_path(df, threshold, min_non_null_abs)
    _new_polars_path(df, threshold, min_non_null_abs)

    legacy_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _legacy_polars_path(df, threshold, min_non_null_abs)
        legacy_times.append(time.perf_counter() - t0)
    new_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _new_polars_path(df, threshold, min_non_null_abs)
        new_times.append(time.perf_counter() - t0)

    leg_med = float(np.median(legacy_times))
    new_med = float(np.median(new_times))
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_runs": n_runs,
        "legacy_wall_seconds": legacy_times,
        "new_wall_seconds": new_times,
        "legacy_median_s": leg_med,
        "new_median_s": new_med,
        "speedup_x": leg_med / new_med if new_med > 0 else float("nan"),
    }


def main() -> dict:
    scenarios = [
        _bench(n_rows=200_000, n_cols=60, n_runs=5),
        _bench(n_rows=50_000, n_cols=120, n_runs=5),
    ]
    out = {"scenarios": scenarios}
    out_path = RESULTS_DIR / "bench_per_col_auto_detect.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for sc in scenarios:
        print(
            f"n_rows={sc['n_rows']:_} n_cols={sc['n_cols']}: "
            f"legacy={sc['legacy_median_s']*1000:.1f}ms new={sc['new_median_s']*1000:.1f}ms "
            f"speedup={sc['speedup_x']:.2f}x"
        )
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
