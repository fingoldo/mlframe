"""Isolated WALL-TIME microbench for the top training-core profile lead:
``feature_drift_report._col_value_counts`` / ``compute_categorical_drift_psi``.

The discovery cProfile (``profile_training_core_hotpath.py``) flagged
``pandas.core.algorithms.value_counts_arraylike`` at ~6.49s tottime (61% of a
10.6s n=2000 hgb suite run), attributed to ``feature_drift_report.py:457
_col_value_counts`` called from ``compute_categorical_drift_psi`` (the
honest-diagnostics drift block, always-on by default).

cProfile inflates deep pandas stacks ~10-13x, so this MUST be confirmed by wall
time before being treated as a real lead. This bench times the EXACT call on the
EXACT synthetic frame at both shapes, warm + multi-iteration, to separate real
cost from attribution noise.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np

import mlframe.training.pipeline  # noqa: F401  native DLL load-order guard (see profile harness)
from mlframe.training.feature_drift_report import (
    _col_value_counts,
    _categorical_columns,
    compute_categorical_drift_psi,
)
from mlframe.training._benchmarks._profile_fuzz_1m import _make_synthetic_frame

_RESULTS_DIR = Path(__file__).resolve().parent / "_results"


def _time_call(fn, *a, iters=20, warmup=3):
    for _ in range(warmup):
        fn(*a)
    best = float("inf")
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*a)
        dt = time.perf_counter() - t0
        best = min(best, dt)
        t.append(dt)
    return {"best_us": best * 1e6, "median_us": float(np.median(t)) * 1e6}


def _time_call_slow(fn, *a):
    """Single-shot timing for object-array columns whose value_counts is seconds-scale
    (warm + multi-iter would blow the 3-min budget; one call is representative)."""
    t0 = time.perf_counter()
    fn(*a)
    dt = time.perf_counter() - t0
    return {"best_us": dt * 1e6, "median_us": dt * 1e6, "single_shot": True}


def _is_object_array_col(df, c) -> bool:
    try:
        return str(df[c].dtype) == "object" and isinstance(df[c].iloc[0], np.ndarray)
    except Exception as exc:  # noqa: BLE001
        logger.debug("_is_object_array_col: probe failed for column %r: %s", c, exc)
        return False


def main():
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"label": "drift-value-counts-microbench", "shapes": []}
    # MEASURED: emb (object-dtype ndarray cells) value_counts is 4.24s wall @ n=2000
    # and DID NOT COMPLETE in 12+ min @ 40k (pandas PyObjectHashTable hashing each
    # ndarray cell -- super-linear). That non-completion IS the result. The bench
    # runs only n=2000 by default; pass larger n on the CLI to re-witness the hang.
    import sys
    _ns = [int(x) for x in sys.argv[1:]] or [2_000]
    for n in _ns:
        df = _make_synthetic_frame("regression", n, seed=11)
        cat_cols = _categorical_columns(df)
        rec = {"n_rows": n, "cat_cols": cat_cols, "per_col": {}}
        # Per-column wall time of _col_value_counts (the flagged hotspot).
        for c in cat_cols:
            try:
                nun = int(df[c].nunique(dropna=False))
            except Exception as e:  # noqa: BLE001  e.g. unhashable ndarray cells (embedding col)
                nun = f"UNHASHABLE:{type(e).__name__}"
            if _is_object_array_col(df, c):
                r = _time_call_slow(_col_value_counts, df, c)
            else:
                r = _time_call(_col_value_counts, df, c, iters=15, warmup=2)
            r["nunique"] = nun
            r["dtype"] = str(df[c].dtype)
            sample0 = df[c].iloc[0]
            r["cell0_type"] = type(sample0).__name__
            rec["per_col"][c] = r
        # Whole-path wall time: compute_categorical_drift_psi(train,val,test).
        # Suite splits train/val/test; mimic with 60/20/20 row slices.
        n0, n1 = int(n * 0.6), int(n * 0.8)
        tr, va, te = df.iloc[:n0], df.iloc[n0:n1], df.iloc[n1:]
        # Single-shot: the whole path runs emb value_counts x3 (train/val/test),
        # which is seconds-to-minutes at 40k -- the bug under measurement.
        whole = _time_call_slow(compute_categorical_drift_psi, tr, va, te)
        rec["compute_categorical_drift_psi"] = whole
        out["shapes"].append(rec)
        print(f"n={n:_} cat_cols={cat_cols}")
        for c, r in rec["per_col"].items():
            print(f"  _col_value_counts[{c}] dtype={r['dtype']} cell0={r['cell0_type']} nunique={r['nunique']} best={r['best_us']:.1f}us median={r['median_us']:.1f}us")
        print(f"  compute_categorical_drift_psi WHOLE best={whole['best_us']:.1f}us median={whole['median_us']:.1f}us")

    p = _RESULTS_DIR / "drift_value_counts_microbench.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return out


if __name__ == "__main__":
    main()
