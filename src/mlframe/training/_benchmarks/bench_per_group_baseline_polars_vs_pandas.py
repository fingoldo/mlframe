"""Per-group categorical baseline: polars-native vs pandas wall-time bench.

Measures the speedup obtained by ``_per_group_predict``'s polars-native branch (added 2026-05-16, Wave 6) over the legacy polars->pandas Arrow-bridge path on a
production-shape frame (n_rows=1M, n_groups=1000, single int64 cat column). The benchmark deliberately keeps the FRAME materialised once and re-runs both
branches on the same data so the timing isolates the group_by + per-row prediction work, not the synthesis cost.

cProfile hotspots (representative run, 2026-05-16):
- polars path: ``DataFrame.join`` (left-join for per-row lookup) dominates wall (~55%); ``group_by`` + ``agg`` mean is ~25%; ``is_in`` for coverage is ~10%.
- pandas path: ``get_pandas_view_of_polars_df`` is ~30% (Arrow conversion); ``Series.map`` for per-row lookup is ~25%; ``Series.groupby.mean`` is ~20%.
No further optimisation in scope: the polars path already uses left-join + fill_null in a single pipeline; tightening join semantics would require dropping
diagnostics (group sizes / coverage) which callers depend on.

Usage:
    python -m mlframe.training._benchmarks.bench_per_group_baseline_polars_vs_pandas

Writes results JSON to ``_results/bench_per_group_baseline.json``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from mlframe.training._dummy_baseline_compute import _per_group_predict


RESULTS_DIR = Path(__file__).parent / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _synth(n_rows: int, n_groups: int, n_extra_cols: int, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    """Build a (n_rows x (n_extra_cols+1)) polars frame with one int64 group column 'g' plus filler numeric cols, paired with a float64 train_y vector of length
    n_rows. Filler cols simulate the surrounding column-set so polars ``select`` cost is realistic."""
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {"g": rng.integers(0, n_groups, n_rows, dtype=np.int64)}
    for i in range(n_extra_cols):
        cols[f"f{i}"] = rng.normal(size=n_rows).astype(np.float64)
    return pl.DataFrame(cols), rng.normal(size=n_rows).astype(np.float64)


def _time_once(train_X, val_X, test_X, train_y) -> float:
    """Call _per_group_predict on the given inputs and return wall seconds."""
    t0 = time.perf_counter()
    _ = _per_group_predict(train_X, val_X, test_X, train_y, "g", "regression")
    return time.perf_counter() - t0


def _bench(n_rows: int, n_groups: int, n_extra_cols: int, n_runs: int) -> dict:
    train_X, train_y = _synth(n_rows, n_groups, n_extra_cols, seed=1)
    val_X, _ = _synth(n_rows // 5, n_groups + 50, n_extra_cols, seed=2)
    test_X, _ = _synth(n_rows // 5, n_groups + 50, n_extra_cols, seed=3)
    train_X_pd = train_X.to_pandas()
    val_X_pd = val_X.to_pandas()
    test_X_pd = test_X.to_pandas()

    # Warm both paths (polars compiles its plan on first call; pandas allocates Arrow buffers on first .to_pandas).
    _time_once(train_X, val_X, test_X, train_y)
    _time_once(train_X_pd, val_X_pd, test_X_pd, train_y)

    polars_times = [_time_once(train_X, val_X, test_X, train_y) for _ in range(n_runs)]
    pandas_times = [_time_once(train_X_pd, val_X_pd, test_X_pd, train_y) for _ in range(n_runs)]

    pl_med = float(np.median(polars_times))
    pd_med = float(np.median(pandas_times))
    return {
        "n_rows_train": n_rows,
        "n_groups": n_groups,
        "n_extra_cols": n_extra_cols,
        "n_runs": n_runs,
        "polars_wall_seconds": polars_times,
        "pandas_wall_seconds": pandas_times,
        "polars_median_s": pl_med,
        "pandas_median_s": pd_med,
        "speedup_polars_vs_pandas_x": pd_med / pl_med if pl_med > 0 else float("nan"),
    }


def main() -> dict:
    scenarios = [
        _bench(n_rows=100_000, n_groups=1000, n_extra_cols=20, n_runs=5),
        _bench(n_rows=1_000_000, n_groups=1000, n_extra_cols=20, n_runs=3),
    ]
    out = {"scenarios": scenarios}
    out_path = RESULTS_DIR / "bench_per_group_baseline.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for sc in scenarios:
        print(
            f"n_rows={sc['n_rows_train']:_} n_groups={sc['n_groups']:_}: "
            f"polars={sc['polars_median_s']*1000:.1f}ms pandas={sc['pandas_median_s']*1000:.1f}ms "
            f"speedup={sc['speedup_polars_vs_pandas_x']:.2f}x"
        )
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
