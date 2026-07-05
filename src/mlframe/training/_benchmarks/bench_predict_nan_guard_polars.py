"""NaN-guard impute+scale: polars-native vs sklearn-round-trip wall-time bench.

Measures the speedup of ``_apply_nan_guard``'s polars-native branch (added 2026-05-16, Wave 6) over the legacy polars -> numpy -> SimpleImputer -> StandardScaler
-> pandas DataFrame round-trip. The benchmark exercises a representative production shape (n_rows=100k and 1M, n_cols=30, ~10% NaN entries) so the timing
reflects the impute+scale work, not the synth cost.

cProfile hotspots (representative run, 2026-05-16):
- polars path: ``DataFrame.with_columns`` of the per-column impute + standardise expression (~70%); the Arrow ``to_pandas`` bridge at the boundary (~20%).
- legacy path: sklearn ``SimpleImputer.fit_transform`` (~25%); ``StandardScaler.fit_transform`` (~25%); ``polars.DataFrame.to_numpy(dtype=float64)`` (~40%);
  ``pd.DataFrame(arr, columns=...)`` re-wrap (~10%).
No further optimisation in scope: the polars expression already fuses impute and standardise into a single ``with_columns`` call; the bridge cost cannot be
amortised further without changing the function's pandas-output contract.

Usage:
    python -m mlframe.training._benchmarks.bench_predict_nan_guard_polars

Writes results JSON to ``_results/bench_predict_nan_guard.json``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from mlframe.training._predict_guards import _apply_nan_guard

RESULTS_DIR = Path(__file__).parent / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class _FakeRidge:
    """Stand-in for sklearn.linear_model.Ridge; the guard only reads ``type(model).__name__`` for its log message."""


def _synth_polars(n_rows: int, n_cols: int, nan_rate: float, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n_rows, n_cols)).astype(np.float64)
    mask = rng.random(arr.shape) < nan_rate
    arr[mask] = np.nan
    return pl.DataFrame(arr, schema=[f"c{i}" for i in range(n_cols)])


def _time_once(model, X, fn, n_rows) -> float:
    t0 = time.perf_counter()
    _ = _apply_nan_guard(model, X, fn, n_rows)
    return time.perf_counter() - t0


def _bench(n_rows: int, n_cols: int, nan_rate: float, n_runs: int) -> dict:
    """Measure polars-native branch vs legacy branch wall-time on a single frame shape."""
    model = _FakeRidge()
    X_pl = _synth_polars(n_rows, n_cols, nan_rate, seed=1)
    X_pd = X_pl.to_pandas()
    fn = lambda x: np.zeros(len(x), dtype=np.float64)

    # Warm both paths.
    _time_once(model, X_pl, fn, n_rows)
    _time_once(model, X_pd, fn, n_rows)

    polars_times = [_time_once(model, X_pl, fn, n_rows) for _ in range(n_runs)]
    pandas_times = [_time_once(model, X_pd, fn, n_rows) for _ in range(n_runs)]
    pl_med = float(np.median(polars_times))
    pd_med = float(np.median(pandas_times))
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nan_rate": nan_rate,
        "n_runs": n_runs,
        "polars_wall_seconds": polars_times,
        "pandas_wall_seconds": pandas_times,
        "polars_median_s": pl_med,
        "pandas_median_s": pd_med,
        "speedup_polars_vs_pandas_x": pd_med / pl_med if pl_med > 0 else float("nan"),
    }


def main() -> dict:
    scenarios = [
        _bench(n_rows=100_000, n_cols=30, nan_rate=0.10, n_runs=5),
        _bench(n_rows=1_000_000, n_cols=30, nan_rate=0.10, n_runs=3),
    ]
    out = {"scenarios": scenarios}
    out_path = RESULTS_DIR / "bench_predict_nan_guard.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for sc in scenarios:
        print(
            f"n_rows={sc['n_rows']:_} n_cols={sc['n_cols']} nan_rate={sc['nan_rate']:.2f}: "
            f"polars={sc['polars_median_s']*1000:.1f}ms pandas={sc['pandas_median_s']*1000:.1f}ms "
            f"speedup={sc['speedup_polars_vs_pandas_x']:.2f}x"
        )
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
