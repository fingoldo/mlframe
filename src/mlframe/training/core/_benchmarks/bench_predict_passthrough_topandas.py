"""TC05 microbench: predict passthrough polars -> pandas column extraction.

Compares two ways of pulling a set of passthrough columns out of a polars
frame into pandas during the predict path:

  * BATCHED  -- ``df.select(cols).to_pandas()`` (one Arrow table, one
    materialise). This is the path the predict code uses.
  * PERCOL  -- ``{c: df.get_column(c).to_pandas() for c in cols}`` then
    ``pd.DataFrame(...)`` (one Arrow Series materialise per column + a
    pandas concat to reassemble). The intuitive "just grab each column"
    alternative.

Verdict (committed as reproducible negative result per CLAUDE.md
REJECTED != DELETED): per-column is REJECTED -- it is ~1.6x slower because
each column pays its own Arrow->numpy dispatch + the pandas frame has to be
reassembled column-by-column, whereas the batched select materialises the
whole sub-table in one threaded pyarrow pass. The batched path is already
what predict uses; this bench is the evidence it is optimal, nothing to
change.

Run: CUDA_VISIBLE_DEVICES="" python bench_predict_passthrough_topandas.py
Output JSON -> _results/predict_passthrough_topandas.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

_RESULTS_DIR = Path(__file__).resolve().parent / "_results"


def _make_frame(n: int, n_cols: int, seed: int = 0) -> pl.DataFrame:
    """Mixed str/numeric frame: half string passthrough cols, half float64."""
    rng = np.random.default_rng(seed)
    data = {}
    n_str = n_cols // 2 + (n_cols % 2)
    n_num = n_cols - n_str
    pool = np.array([f"id_{k:04d}" for k in range(2000)])
    for j in range(n_str):
        data[f"s{j}"] = pool[rng.integers(0, pool.size, n)]
    for j in range(n_num):
        data[f"x{j}"] = rng.standard_normal(n)
    return pl.DataFrame(data)


def _batched(df: pl.DataFrame, cols: list[str]):
    return df.select(cols).to_pandas()


def _percol(df: pl.DataFrame, cols: list[str]):
    import pandas as pd

    return pd.DataFrame({c: df.get_column(c).to_pandas() for c in cols})


def _time_it(fn, df, cols, *, iters: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn(df, cols)
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(df, cols)
        best = min(best, time.perf_counter() - t0)
    return best


def _identical(df: pl.DataFrame, cols: list[str]) -> bool:
    a = _batched(df, cols)
    b = _percol(df, cols)
    if list(a.columns) != list(b.columns):
        return False
    for c in cols:
        if a[c].dtype != b[c].dtype:
            return False
        if a[c].dtype.kind == "f":
            if not np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True):
                return False
        elif not (a[c].to_numpy() == b[c].to_numpy()).all():
            return False
    return True


def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for n in (100_000, 500_000):
        iters = 20 if n <= 100_000 else 8
        for n_cols in (3, 8):
            df = _make_frame(n, n_cols)
            cols = df.columns
            ident = _identical(df, cols)
            t_batched = _time_it(_batched, df, cols, iters=iters)
            t_percol = _time_it(_percol, df, cols, iters=iters)
            ratio = t_percol / t_batched
            row = {
                "n": n,
                "n_cols": n_cols,
                "iters": iters,
                "batched_ms": round(t_batched * 1e3, 4),
                "percol_ms": round(t_percol * 1e3, 4),
                "percol_over_batched": round(ratio, 3),
                "batched_faster": t_batched < t_percol,
                "bit_identical": ident,
            }
            results.append(row)
            print(
                f"n={n:>7} cols={n_cols}: batched={row['batched_ms']:>8.3f}ms "
                f"percol={row['percol_ms']:>8.3f}ms  percol/batched={ratio:.2f}x "
                f"identical={ident}"
            )

    ratios = [r["percol_over_batched"] for r in results]
    out = {
        "bench": "TC05_predict_passthrough_topandas",
        "verdict": "PERCOL_REJECTED",
        "summary": (
            "batched select().to_pandas() is faster on every cell; per-column "
            "get_column().to_pandas() is ~1.6x slower (reassembly + per-col dispatch). "
            "Batched path is already what predict uses."
        ),
        "percol_over_batched_min": round(min(ratios), 3),
        "percol_over_batched_max": round(max(ratios), 3),
        "all_batched_faster": all(r["batched_faster"] for r in results),
        "all_bit_identical": all(r["bit_identical"] for r in results),
        "env": {"polars": pl.__version__, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "results": results,
    }
    out_path = _RESULTS_DIR / "predict_passthrough_topandas.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
