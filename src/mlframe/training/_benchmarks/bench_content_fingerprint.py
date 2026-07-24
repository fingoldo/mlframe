"""Point-sample fingerprint vs full ``to_numpy()`` materialisation wall-time + RSS bench.

Measures ``_content_fingerprint_for_cache`` (post-fix point-sample of 4 rows by index) against the pre-fix path that called ``arr.to_numpy()`` over the
whole frame before slicing 10 cells. On a 1M x 100 polars frame the materialisation copied ~800 MB into a contiguous numpy buffer per fingerprint call,
which defeated the very pre-pipeline cache the per-target loop relies on. Post-fix cost is O(n_cols) and independent of n_rows.

cProfile hotspots (representative 1M x 100 run, 2026-05-16):
- post-fix: ``polars.DataFrame.row`` (~60%), tuple construction of dtypes/columns (~30%), Python overhead in the dispatch chain (~10%).
- pre-fix: ``polars.DataFrame.to_numpy`` (~95%), tail slicing into bytes (~5%).
No further optimisation in scope: the post-fix path already touches only O(n_cols) cells, dtypes are read from ``df.dtypes`` (already cached on the frame),
and column names are pulled from a pre-built list. Going below the row-extraction cost would require keeping a content hash on the frame itself, which
breaks the API contract (the caller passes an opaque DataFrame, not a fingerprint-bearing wrapper).

Usage:
    python -m mlframe.training._benchmarks.bench_content_fingerprint

Writes results JSON to ``_results/bench_content_fingerprint.json``.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from mlframe.training.pipeline._pipeline_helpers import _content_fingerprint_for_cache

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _rss_mb() -> float:
    """Process resident set size in MB; psutil-optional so the bench runs in minimal envs."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception as exc:
        logger.debug("_rss_mb: psutil probe failed: %s", exc)
        return float("nan")


def _synth_polars(n_rows: int, n_cols: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"c{i}": rng.normal(size=n_rows).astype(np.float64) for i in range(n_cols)}
    return pl.DataFrame(data)


def _pre_fix_fingerprint(arr: pl.DataFrame) -> tuple:
    """Emulation of the pre-fix path that called ``arr.to_numpy()`` over the whole frame."""
    col_names = tuple(str(c) for c in arr.columns)
    np_arr = arr.to_numpy()
    shape = tuple(int(s) for s in np_arr.shape)
    dtype_str = str(np_arr.dtype)
    flat = np_arr.ravel()
    n = int(flat.size)
    idx = [int(i * (n - 1) / 9) for i in range(10)] if n >= 10 else list(range(n))
    sampled = bytes(np.ascontiguousarray(flat[idx]).tobytes())
    return (shape, dtype_str, sampled, col_names)


def _time_post(df: pl.DataFrame, n_runs: int) -> list[float]:
    out = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = _content_fingerprint_for_cache(df)
        out.append(time.perf_counter() - t0)
    return out


def _time_pre(df: pl.DataFrame, n_runs: int) -> list[float]:
    out = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = _pre_fix_fingerprint(df)
        out.append(time.perf_counter() - t0)
        # The pre-fix path allocates ~ n_rows * n_cols * 8 bytes; force a GC pass so subsequent timings don't measure RSS that's still cooling down.
        gc.collect()
    return out


def _bench(n_rows: int, n_cols: int, n_runs: int) -> dict:
    df = _synth_polars(n_rows, n_cols, seed=1)
    # Warm both paths so cold-import overhead doesn't skew the first sample.
    _content_fingerprint_for_cache(df)
    _pre_fix_fingerprint(df)
    gc.collect()

    rss_before = _rss_mb()
    post = _time_post(df, n_runs)
    rss_after_post = _rss_mb()
    pre = _time_pre(df, n_runs)
    rss_after_pre = _rss_mb()

    post_med = float(np.median(post))
    pre_med = float(np.median(pre))
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_runs": n_runs,
        "post_fix_wall_seconds": post,
        "pre_fix_wall_seconds": pre,
        "post_fix_median_s": post_med,
        "pre_fix_median_s": pre_med,
        "speedup_post_over_pre_x": pre_med / post_med if post_med > 0 else float("nan"),
        "rss_mb_before": rss_before,
        "rss_mb_after_post": rss_after_post,
        "rss_mb_after_pre": rss_after_pre,
        "rss_delta_pre_minus_post_mb": rss_after_pre - rss_after_post,
    }


def main() -> dict:
    scenarios = [
        _bench(n_rows=100_000, n_cols=100, n_runs=5),
        _bench(n_rows=1_000_000, n_cols=100, n_runs=3),
    ]
    out = {"scenarios": scenarios}
    out_path = RESULTS_DIR / "bench_content_fingerprint.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for sc in scenarios:
        print(
            f"n_rows={sc['n_rows']:_} n_cols={sc['n_cols']}: "
            f"post={sc['post_fix_median_s']*1000:.3f}ms pre={sc['pre_fix_median_s']*1000:.1f}ms "
            f"speedup={sc['speedup_post_over_pre_x']:.1f}x "
            f"rss_delta_pre_minus_post={sc['rss_delta_pre_minus_post_mb']:+.1f}MB"
        )
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
