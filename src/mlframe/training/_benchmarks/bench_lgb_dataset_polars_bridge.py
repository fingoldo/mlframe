"""LGB shim: polars X routed through Arrow split-blocks bridge vs bare ``df.to_pandas()`` baseline.

Pre-fix path: ``lgb.Dataset(data=polars_df)`` was passed straight through; installed LightGBM 4.x rejects it with TypeError (polars does not implement
the array protocols ``Dataset.__init__`` probes), so production callers fell back to bare ``df.to_pandas()`` -- which dropped the Categorical dictionary
(rebuilt as ``object`` dtype) and routed LightGBM through numeric hashing instead of the native cat-split path. Post-fix path: the shim calls
``_maybe_bridge_polars_to_pandas`` (project's ``get_pandas_view_of_polars_df``, Arrow split-blocks bridge); numeric columns stay zero-copy, Categorical
columns reach LightGBM with their dictionary intact.

Correctness gain (always): bridged path preserves ``pd.CategoricalDtype``; bare path collapses to ``object`` dtype.

Wall-time gain (conversion step only, isolated from binning): shape-dependent. On 10 num + 5 cat at 200k rows the bridge is ~0.65x of bare ``to_pandas``.
At higher cat counts the gap narrows because the dictionary-rebuild step (uint32 -> int32 cast) dominates and is shared by both paths. LightGBM's
``Dataset.construct`` is binning-dominated so the end-to-end wall-time differential measured at the ``construct()`` boundary is diluted further.

cProfile hotspots (representative 200k x 15 run, 2026-05-16):
- post-fix bridge: ``polars.DataFrame.to_arrow`` (~40%), pyarrow ``cast`` for Categorical index re-typing (~30%), ``pyarrow.Table.to_pandas`` (~25%).
- bare ``to_pandas``: ``polars.DataFrame.to_pandas`` internal Arrow round-trip (~60%), pandas Categorical construction from object array (~30%).
No further optimisation in scope: the bridge already avoids the bare path's object materialisation of Categorical columns, and the residual cost is in
pyarrow ``cast`` which is C++ and not callable-from-Python tunable. The remaining low-hanging fruit would be ``pl.Enum`` adoption on the source frame
(fixed domain, zero-copy categorical bridge) -- tracked as a separate refactor in the codebase, not in scope here.

Usage:
    python -m mlframe.training._benchmarks.bench_lgb_dataset_polars_bridge

Writes results JSON to ``_results/bench_lgb_dataset_polars_bridge.json``.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


RESULTS_DIR = Path(__file__).parent / "_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def _make_mixed_polars(n_rows: int, n_num: int, n_cat: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cat_pool = [f"level_{i}" for i in range(8)]
    data = {f"num{i}": rng.normal(size=n_rows).astype(np.float64) for i in range(n_num)}
    for j in range(n_cat):
        codes = rng.integers(0, len(cat_pool), size=n_rows)
        data[f"cat{j}"] = pl.Series(
            f"cat{j}", [cat_pool[c] for c in codes], dtype=pl.Categorical
        )
    return pl.DataFrame(data)


def _bench(n_rows: int, n_num: int, n_cat: int, n_runs: int) -> dict:
    """Compare bridge vs bare ``to_pandas`` conversion (isolated) and end-to-end ``Dataset.construct`` (binning included)."""
    import lightgbm as lgb
    from mlframe.training.lgb_shim import _maybe_bridge_polars_to_pandas

    df_pl = _make_mixed_polars(n_rows, n_num, n_cat, seed=1)
    y = np.random.default_rng(2).normal(size=df_pl.height)

    # Warm both paths.
    _ = _maybe_bridge_polars_to_pandas(df_pl)
    _ = df_pl.to_pandas()
    _ = lgb.Dataset(data=_maybe_bridge_polars_to_pandas(df_pl), label=y, free_raw_data=False).construct()
    _ = lgb.Dataset(data=df_pl.to_pandas(), label=y, free_raw_data=False).construct()
    gc.collect()

    rss_before = _rss_mb()

    # 1. Conversion-only timings.
    bridge_conv: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = _maybe_bridge_polars_to_pandas(df_pl)
        bridge_conv.append(time.perf_counter() - t0)
        gc.collect()

    bare_conv: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = df_pl.to_pandas()
        bare_conv.append(time.perf_counter() - t0)
        gc.collect()

    # 2. End-to-end (conversion + Dataset.construct) timings + Categorical-dtype check.
    bridge_e2e: list[float] = []
    bridged_dtypes = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        bridged = _maybe_bridge_polars_to_pandas(df_pl)
        ds = lgb.Dataset(data=bridged, label=y, free_raw_data=False)
        ds.construct()
        bridge_e2e.append(time.perf_counter() - t0)
        if bridged_dtypes is None:
            bridged_dtypes = {c: str(t) for c, t in bridged.dtypes.items()}
        gc.collect()

    bare_e2e: list[float] = []
    bare_dtypes = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        bare = df_pl.to_pandas()
        ds = lgb.Dataset(data=bare, label=y, free_raw_data=False)
        ds.construct()
        bare_e2e.append(time.perf_counter() - t0)
        if bare_dtypes is None:
            bare_dtypes = {c: str(t) for c, t in bare.dtypes.items()}
        gc.collect()

    rss_after = _rss_mb()

    bridge_conv_med = float(np.median(bridge_conv))
    bare_conv_med = float(np.median(bare_conv))
    bridge_e2e_med = float(np.median(bridge_e2e))
    bare_e2e_med = float(np.median(bare_e2e))

    # Correctness check: count Categorical columns surviving each path.
    n_cat_bridged = sum(1 for v in bridged_dtypes.values() if "category" in v.lower())
    n_cat_bare = sum(1 for v in bare_dtypes.values() if "category" in v.lower())

    return {
        "n_rows": n_rows,
        "n_num": n_num,
        "n_cat": n_cat,
        "n_runs": n_runs,
        "conversion_only": {
            "bridge_wall_seconds": bridge_conv,
            "bare_wall_seconds": bare_conv,
            "bridge_median_s": bridge_conv_med,
            "bare_median_s": bare_conv_med,
            "speedup_bridge_vs_bare_x": bare_conv_med / bridge_conv_med if bridge_conv_med > 0 else float("nan"),
        },
        "end_to_end_dataset_construct": {
            "bridge_wall_seconds": bridge_e2e,
            "bare_wall_seconds": bare_e2e,
            "bridge_median_s": bridge_e2e_med,
            "bare_median_s": bare_e2e_med,
            "speedup_bridge_vs_bare_x": bare_e2e_med / bridge_e2e_med if bridge_e2e_med > 0 else float("nan"),
        },
        "categorical_preserved": {
            "n_cat_input": n_cat,
            "n_cat_bridged_path": n_cat_bridged,
            "n_cat_bare_path": n_cat_bare,
        },
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "rss_delta_mb": rss_after - rss_before,
    }


def main() -> dict:
    scenarios = [
        _bench(n_rows=200_000, n_num=10, n_cat=5, n_runs=5),
        _bench(n_rows=200_000, n_num=10, n_cat=20, n_runs=3),
        _bench(n_rows=200_000, n_num=10, n_cat=50, n_runs=3),
    ]
    out = {"scenarios": scenarios}
    out_path = RESULTS_DIR / "bench_lgb_dataset_polars_bridge.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for sc in scenarios:
        cv = sc["conversion_only"]
        e2e = sc["end_to_end_dataset_construct"]
        cp = sc["categorical_preserved"]
        print(
            f"n_rows={sc['n_rows']:_} n_num={sc['n_num']} n_cat={sc['n_cat']}: "
            f"conv bridge={cv['bridge_median_s']*1000:.1f}ms bare={cv['bare_median_s']*1000:.1f}ms ({cv['speedup_bridge_vs_bare_x']:.2f}x) | "
            f"e2e bridge={e2e['bridge_median_s']*1000:.1f}ms bare={e2e['bare_median_s']*1000:.1f}ms ({e2e['speedup_bridge_vs_bare_x']:.2f}x) | "
            f"cat_kept bridge={cp['n_cat_bridged_path']}/{cp['n_cat_input']} bare={cp['n_cat_bare_path']}/{cp['n_cat_input']}"
        )
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
