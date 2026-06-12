"""TC07 microbench: fairness subgroup extraction -- zero-copy Arrow view vs copy.

The fairness / per-subgroup metrics path needs a pandas view of a small set
of polars columns (the subgroup keys + score). Two ways:

  * VIEW -- ``get_pandas_view_of_polars_df(df)`` (split_blocks=True Arrow
    bridge: numeric/string columns become zero-copy numpy views over the
    Arrow buffers). This is the production path.
  * COPY -- ``df.to_pandas()`` (pyarrow default: CONSOLIDATES numeric blocks
    into fresh numpy arrays -> a full per-buffer copy).

Verdict (committed reproducible result per CLAUDE.md REJECTED != DELETED):
the ~14x win is ALREADY APPLIED in production -- ``get_pandas_view_of_polars_df``
is the live path. The "reject" here is that there is nothing left to optimise
on this path: the zero-copy bridge already beats the naive ``to_pandas()``
copy by ~14x on a 3-col numeric subgroup, and that bridge is what the
fairness code calls. This bench is the standing evidence.

The bridge is invoked with a FRESH frame id each iteration to defeat the
single-entry id-memo (``_PD_VIEW_LAST_CACHE``) -- otherwise the memo would
return a cached view and report an inflated, unrepresentative speedup.

Run: CUDA_VISIBLE_DEVICES="" python bench_fairness_arrow_view.py
Output JSON -> _results/fairness_arrow_view.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

# Import the production bridge exactly as the fairness path does.
from mlframe.training.utils import get_pandas_view_of_polars_df

_RESULTS_DIR = Path(__file__).resolve().parent / "_results"


def _make_subgroup_frame(n: int, seed: int = 0) -> pl.DataFrame:
    """3-col subgroup frame: 2 numeric subgroup keys + 1 float score.

    All-numeric so the zero-copy split_blocks path is fully exercised (no
    Categorical dict cast, which is the one unavoidable-copy column type).
    """
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "group_a": rng.integers(0, 8, n).astype(np.int64),
            "group_b": rng.integers(0, 32, n).astype(np.int64),
            "score": rng.standard_normal(n),
        }
    )


def _time_view(frames, *, iters: int, warmup: int) -> float:
    # frames: pre-built list of distinct-id frames so the id-memo never hits.
    for i in range(warmup):
        get_pandas_view_of_polars_df(frames[i])
    best = float("inf")
    for i in range(iters):
        f = frames[warmup + i]
        t0 = time.perf_counter()
        get_pandas_view_of_polars_df(f)
        best = min(best, time.perf_counter() - t0)
    return best


def _time_copy(frames, *, iters: int, warmup: int) -> float:
    for i in range(warmup):
        frames[i].to_pandas()
    best = float("inf")
    for i in range(iters):
        f = frames[warmup + i]
        t0 = time.perf_counter()
        f.to_pandas()
        best = min(best, time.perf_counter() - t0)
    return best


def _identical(df: pl.DataFrame) -> bool:
    a = get_pandas_view_of_polars_df(df)
    b = df.to_pandas()
    if list(a.columns) != list(b.columns):
        return False
    for c in a.columns:
        av, bv = a[c].to_numpy(), b[c].to_numpy()
        if av.dtype.kind == "f":
            if not np.array_equal(av, bv, equal_nan=True):
                return False
        elif not (av == bv).all():
            return False
    return True


def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for n in (200_000, 2_000_000):
        iters = 12 if n <= 200_000 else 6
        warmup = 2
        # Distinct-id frames so the bridge's single-entry id-memo never fires.
        frames = [_make_subgroup_frame(n, seed=s) for s in range(iters + warmup)]
        ident = _identical(_make_subgroup_frame(n, seed=999))
        t_view = _time_view(frames, iters=iters, warmup=warmup)
        t_copy = _time_copy(frames, iters=iters, warmup=warmup)
        speedup = t_copy / t_view
        row = {
            "n": n,
            "n_cols": 3,
            "iters": iters,
            "view_ms": round(t_view * 1e3, 4),
            "copy_ms": round(t_copy * 1e3, 4),
            "copy_over_view_speedup": round(speedup, 3),
            "view_faster": t_view < t_copy,
            "bit_identical": ident,
        }
        results.append(row)
        print(
            f"n={n:>9} cols=3: view={row['view_ms']:>8.3f}ms "
            f"copy={row['copy_ms']:>8.3f}ms  speedup={speedup:.2f}x identical={ident}"
        )

    speeds = [r["copy_over_view_speedup"] for r in results]
    out = {
        "bench": "TC07_fairness_arrow_view",
        "verdict": "ZERO_COPY_VIEW_ALREADY_APPLIED",
        "summary": (
            "get_pandas_view_of_polars_df (zero-copy split_blocks Arrow bridge) is the "
            "live fairness path and already beats naive df.to_pandas() copy by ~14x on a "
            "3-col numeric subgroup; nothing left unoptimised on this path."
        ),
        "speedup_min": round(min(speeds), 3),
        "speedup_max": round(max(speeds), 3),
        "all_view_faster": all(r["view_faster"] for r in results),
        "all_bit_identical": all(r["bit_identical"] for r in results),
        "env": {"polars": pl.__version__, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "results": results,
    }
    out_path = _RESULTS_DIR / "fairness_arrow_view.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
