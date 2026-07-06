"""Bench: pandas constant-numeric-column detection in ``remove_constant_columns``.

OLD path (pre 2026-06-23): per-column ``Series.min() == Series.max()`` (two pandas
reductions, each with per-call dispatch + nan-masking overhead) followed by a THIRD
``Series.isna().all()`` scan for the all-NaN case.

NEW path: one numpy ``nanmin``/``nanmax`` pass over the raw ndarray per column;
``nanmin == nanmax`` catches constants, ``(nanmin is NaN and nanmax is NaN)`` catches
all-NaN columns -- folding the separate ``isna().all()`` pass into the same two
reductions. Bit-identical drop set (see
tests/training/test_remove_constant_columns_parity.py::
test_fused_numeric_constant_detection_matches_old_reference).

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training._benchmarks.bench_remove_constant_columns_pandas_nanminmax

Reference result (200k x 200, 10% const / 10% all-NaN / 80% varying, best-of-7):
    OLD 170.8 ms -> NEW 43.1 ms  (3.96x), identical drop set.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd


def _make_df(n: int = 200_000, ncols: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {}
    for i in range(ncols):
        if i % 10 == 0:
            data[f"c{i}"] = np.full(n, 3.0)  # constant
        elif i % 10 == 1:
            data[f"c{i}"] = np.full(n, np.nan)  # all-NaN
        else:
            data[f"c{i}"] = rng.standard_normal(n)  # varying
    return pd.DataFrame(data)


def _old(df: pd.DataFrame) -> set:
    numeric_cols = df.select_dtypes(include="number").columns
    constant = [c for c in numeric_cols if df[c].min() == df[c].max()]
    all_nan = [c for c in numeric_cols if c not in constant and df[c].isna().all()]
    return set(constant) | set(all_nan)


def _new(df: pd.DataFrame) -> set:
    numeric_cols = df.select_dtypes(include="number").columns
    out = []
    for col in numeric_cols:
        arr = df[col].to_numpy()
        if arr.size == 0:
            out.append(col)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
        if mn == mx or (mn != mn and mx != mx):
            out.append(col)
    return set(out)


def _best(fn, df, n=7) -> float:
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn(df)
        ts.append(time.perf_counter() - t)
    return min(ts)


def main() -> None:
    df = _make_df()
    assert _old(df) == _new(df), "drop set diverged"  # nosec B101 - internal invariant check in src/mlframe/training/_benchmarks, not reachable with untrusted input
    old_ms = _best(_old, df) * 1000
    new_ms = _best(_new, df) * 1000
    print(f"OLD {old_ms:.1f} ms")
    print(f"NEW {new_ms:.1f} ms")
    print(f"speedup {old_ms / new_ms:.2f}x")
    print("identity OK")


if __name__ == "__main__":
    main()
