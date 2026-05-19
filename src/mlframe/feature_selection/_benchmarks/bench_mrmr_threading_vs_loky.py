"""Benchmark: MRMR with joblib threading backend vs. the legacy loky backend.

Pre-fix (iter-371): on a 1M-row cb multiclass + MRMR + binary=medium combo the
joblib loky backend memmap'd the data set per worker, hit the Windows paging
file ceiling at ~3GB RAM, and the loky resource tracker failed to clean up the
dangling joblib_memmapping_folder temp files (WinError 1455).

Post-fix: MRMR's default ``parallel_kwargs`` flips to
``dict(max_nbytes=MAX_JOBLIB_NBYTES, backend="threading")`` so workers run in
the parent process via threads, share the dataset arrays zero-copy, and rely
on numba's GIL-releasing kernels for genuine CPU-parallel work.

Run:
    PYTHONPATH=src D:/ProgramData/anaconda3/python.exe \
        -m mlframe.feature_selection._benchmarks.bench_mrmr_threading_vs_loky \
        --n-rows 500000

Knob defaults match the iter-371 axes (n_rows=500_000, n_features=12,
fe_max_steps=1, fe_ntop_features=5, fe_npermutations=10, binary preset
"medium"). Smaller n_rows runs in less than a minute; the full 1M-row run
takes 5-15 minutes depending on backend and CPU.
"""
from __future__ import annotations

import argparse
import gc
import os
import time
import tracemalloc

import numpy as np
import pandas as pd


def _build_frame(n_rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n_features = 12
    cols = {f"num{i}": rng.standard_normal(n_rows).astype(np.float64) for i in range(n_features)}
    # Target: nonlinear combination of two features + noise; multiclass-ish.
    y_raw = (
        2.0 * cols["num0"] - 1.5 * cols["num1"] * cols["num2"]
        + 0.5 * np.sin(cols["num3"])
        + 0.3 * rng.standard_normal(n_rows)
    )
    # Discretise to 3 classes via quantiles so MI computation has a sensible
    # joint histogram (avoids the all-continuous numba paths that don't
    # exercise the joblib outer loop the bench is measuring).
    edges = np.quantile(y_raw, [0.33, 0.66])
    y = pd.Series(np.digitize(y_raw, edges).astype(np.int32))
    df = pd.DataFrame(cols)
    return df, y


def _peak_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def _run_one(df: pd.DataFrame, y: pd.Series, backend: str, n_jobs: int, verbose: int) -> dict:
    from mlframe.feature_selection.filters.mrmr import MRMR

    parallel_kwargs = dict(max_nbytes=1_000, backend=backend)
    selector = MRMR(
        fe_max_steps=1,
        fe_ntop_features=5,
        fe_npermutations=10,
        fe_unary_preset="minimal",
        fe_binary_preset="medium",
        n_jobs=n_jobs,
        parallel_kwargs=parallel_kwargs,
        verbose=verbose,
        random_seed=11,
    )
    gc.collect()
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    selector.fit(df, y)
    elapsed = time.perf_counter() - t0
    rss_after = _peak_rss_mb()
    n_selected = int(getattr(selector, "support_", np.empty(0)).sum()) if hasattr(selector, "support_") else 0
    return {
        "backend": backend,
        "wall_s": round(elapsed, 2),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "rss_delta_mb": round(rss_after - rss_before, 1) if not np.isnan(rss_before) else float("nan"),
        "n_selected": n_selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    print(f"=== MRMR threading-vs-loky bench (n_rows={args.n_rows:_}, n_jobs={args.n_jobs}) ===")
    df, y = _build_frame(args.n_rows, args.seed)
    print(f"frame: {df.shape}, target: {y.shape} (3 classes)")

    print("\n--- run 1: backend=threading n_jobs=1 (sequential baseline) ---")
    seq = _run_one(df, y, backend="threading", n_jobs=1, verbose=args.verbose)
    print(seq)

    print(f"\n--- run 2: backend=threading n_jobs={args.n_jobs} (post-fix default) ---")
    threading = _run_one(df, y, backend="threading", n_jobs=args.n_jobs, verbose=args.verbose)
    print(threading)

    print("\n--- run 3: backend=loky n_jobs={n} (legacy default — may break on env w/ pickle issues) ---")
    try:
        loky = _run_one(df, y, backend="loky", n_jobs=args.n_jobs, verbose=args.verbose)
        print(loky)
    except Exception as exc:
        print(f"loky run FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        loky = None

    print("\n=== Speedup summary ===")
    speedup_thread = seq["wall_s"] / max(1e-9, threading["wall_s"])
    print(f"threading parallel speedup over seq: {seq['wall_s']}s -> {threading['wall_s']}s = {speedup_thread:.2f}x")
    if loky:
        rss_save = loky["rss_after_mb"] - threading["rss_after_mb"]
        speedup_vs_loky = loky["wall_s"] / max(1e-9, threading["wall_s"])
        print(f"vs loky: {loky['wall_s']}s -> {threading['wall_s']}s = {speedup_vs_loky:.2f}x  (RSS save: {rss_save:.1f}MB)")
    else:
        print("loky baseline unavailable on this env; speedup vs seq is the only safe comparison")


if __name__ == "__main__":
    main()
