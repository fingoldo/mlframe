"""Bench: lazy vs eager prebinning of a polars feature frame -- peak RAM + wall.

Discovery's bin-estimator MI path needs only the int16/int32 CODE matrix; the
float32 (n, F) plane that ``_build_feature_matrix`` materialises feeds nothing
but the prebinning itself (when dedup is off). ``_prebin_feature_columns_lazy``
pulls + bins ONE polars column at a time so the float plane is never built --
peak extra RAM drops from ``(n, F)`` float32 to one column.

This bench measures, on a polars frame (n=200k, F=100):

* tracemalloc PEAK allocated bytes for the EAGER path
  (``_build_feature_matrix`` -> ``_prebin_feature_columns``) vs the LAZY path
  (``_prebin_feature_columns_lazy``), and
* the wall-time of each (lazy adds per-column polars extraction overhead; we
  report it honestly -- a memory win can cost a little wall).

Run::

    CUDA_VISIBLE_DEVICES="" python -m \
        mlframe.training.composite.discovery._benchmarks.bench_lazy_prebin_memory

Honest-verdict policy (CLAUDE.md "REJECTED != DELETED"): if lazy is net-worse on
wall with no real memory win at the tested size, the path stays size-gated where
it helps and this bench documents the crossover. The lazy code is NOT deleted.
"""
from __future__ import annotations

import gc
import json
import os
import tracemalloc
from pathlib import Path
from timeit import default_timer as timer

import numpy as np

from ..screening import _prebin_feature_columns, _prebin_feature_columns_lazy


def _make_frame(n: int, f: int, seed: int = 0):
    import polars as pl

    rng = np.random.default_rng(seed)
    data = {f"c{j}": rng.standard_normal(n).astype(np.float32) for j in range(f)}
    # Sprinkle NaN into a few columns so both paths exercise the nanquantile arm.
    for j in (3, 17, 55):
        if j < f:
            data[f"c{j}"][rng.integers(0, n, n // 50)] = np.nan
    return pl.DataFrame(data), list(data.keys())


def _build_eager_matrix(df, cols, rows):
    # Mirror discovery._build_feature_matrix without importing the class.
    from ..screening import _extract_column_array

    return np.column_stack([_extract_column_array(df, c, rows=rows) for c in cols])


def _peak_and_wall_eager(df, cols, rows, nbins):
    gc.collect()
    tracemalloc.start()
    t0 = timer()
    mat = _build_eager_matrix(df, cols, rows)
    codes = _prebin_feature_columns(mat, nbins=nbins)
    wall = timer() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return codes, peak, wall


def _peak_and_wall_lazy(df, cols, rows, nbins):
    gc.collect()
    tracemalloc.start()
    t0 = timer()
    codes = _prebin_feature_columns_lazy(df, cols, rows, nbins=nbins)
    wall = timer() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return codes, peak, wall


def run(n: int = 200_000, f: int = 100, nbins: int = 16, reps: int = 3) -> dict:
    df, cols = _make_frame(n, f)
    rows = np.arange(n)  # screen sample == full frame at the tested size.

    # Warm caches once (polars lazy schema, numpy) so the timed reps are steady.
    _prebin_feature_columns_lazy(df, cols[:2], rows[:1000], nbins=nbins)

    eager_peaks, eager_walls = [], []
    lazy_peaks, lazy_walls = [], []
    codes_e = codes_l = None
    for _ in range(reps):
        codes_e, pe, we = _peak_and_wall_eager(df, cols, rows, nbins)
        eager_peaks.append(pe)
        eager_walls.append(we)
        codes_l, pl_, wl = _peak_and_wall_lazy(df, cols, rows, nbins)
        lazy_peaks.append(pl_)
        lazy_walls.append(wl)

    bit_identical = bool(np.array_equal(codes_e, codes_l))
    eager_peak = min(eager_peaks)
    lazy_peak = min(lazy_peaks)
    eager_wall = min(eager_walls)
    lazy_wall = min(lazy_walls)
    float_plane_bytes = n * f * 4  # what the eager path materialises (float32).

    result = {
        "n": n,
        "F": f,
        "nbins": nbins,
        "reps": reps,
        "bit_identical": bit_identical,
        "eager_peak_mb": round(eager_peak / 1024 ** 2, 1),
        "lazy_peak_mb": round(lazy_peak / 1024 ** 2, 1),
        "peak_ratio_eager_over_lazy": round(eager_peak / max(lazy_peak, 1), 2),
        "peak_saved_mb": round((eager_peak - lazy_peak) / 1024 ** 2, 1),
        "float_plane_mb": round(float_plane_bytes / 1024 ** 2, 1),
        "eager_wall_s": round(eager_wall, 4),
        "lazy_wall_s": round(lazy_wall, 4),
        "wall_ratio_lazy_over_eager": round(lazy_wall / max(eager_wall, 1e-9), 3),
    }
    return result


def main() -> None:
    res = run()
    out_dir = Path(__file__).resolve().parent / "_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lazy_prebin_memory.json"
    out_path.write_text(json.dumps(res, indent=2, sort_keys=True))
    print(json.dumps(res, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")
    saved = res["peak_saved_mb"]
    wr = res["wall_ratio_lazy_over_eager"]
    print(
        f"\nVERDICT: lazy peak {res['lazy_peak_mb']} MB vs eager "
        f"{res['eager_peak_mb']} MB (saved {saved} MB, "
        f"{res['peak_ratio_eager_over_lazy']}x); wall {wr}x of eager; "
        f"bit_identical={res['bit_identical']}."
    )


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
