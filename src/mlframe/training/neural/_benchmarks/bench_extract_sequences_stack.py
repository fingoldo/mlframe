"""Bench for extract_sequences row-stacking hotspot (training/neural/_recurrent_sequences.py).

extract_sequences turns a polars frame of list-columns (the light-curve layout:
columns mjd/mag/magerr/norm, each cell a per-observation list) into a python list
of ``(seq_len, n_cols)`` float32 arrays -- one ragged array per row. The current
body does, per row, ``np.stack([col_arrays[j][i] for j in range(k)], axis=-1)`` --
n_rows separate small ``np.stack`` calls, each building a python list of k slices
and allocating.

This bench compares the current per-row ``np.stack`` against a variant that, for
the (overwhelmingly common) equal-length-per-row case, transposes once: convert each
column's full list-of-lists, then for each row build the (seq_len, k) array by writing
columns into a preallocated buffer via fancy index -- avoiding the per-row python list
construction inside np.stack.

Run: CUDA_VISIBLE_DEVICES="" python -m mlframe.training.neural._benchmarks.bench_extract_sequences_stack
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl


def _make_df(n_rows: int, seq_len: int, columns):
    rng = np.random.default_rng(0)
    data = {}
    for c in columns:
        # equal length per row (the aligned light-curve case)
        data[c] = [rng.standard_normal(seq_len).tolist() for _ in range(n_rows)]
    return pl.DataFrame(data)


def _old(df, columns):
    col_arrays = [
        [np.asarray(v, dtype=np.float32) for v in df[col].to_list()]
        for col in columns
    ]
    return [
        np.stack([col_arrays[j][i] for j in range(len(columns))], axis=-1)
        for i in range(len(df))
    ]


def _new(df, columns):
    k = len(columns)
    n_rows = len(df)
    col_arrays = [
        [np.asarray(v, dtype=np.float32) for v in df[col].to_list()]
        for col in columns
    ]
    result = []
    for i in range(n_rows):
        seq_len = col_arrays[0][i].shape[0]
        buf = np.empty((seq_len, k), dtype=np.float32)
        for j in range(k):
            buf[:, j] = col_arrays[j][i]
        result.append(buf)
    return result


def _new2(df, columns):
    """Equal-length fast path: stack all rows of each column into one 2-D array
    via a single np.asarray, then split row-wise. Avoids per-cell np.asarray."""
    k = len(columns)
    n_rows = len(df)
    # one np.asarray per column over the whole list-of-lists -> (n_rows, seq_len).
    # Ragged (unequal per-row lengths) raises ValueError on modern numpy -> fall back.
    try:
        col_mats = [np.asarray(df[col].to_list(), dtype=np.float32) for col in columns]
        if any(m.ndim != 2 for m in col_mats):
            raise ValueError("ragged")
    except ValueError:
        col_arrays = [
            [np.asarray(v, dtype=np.float32) for v in df[col].to_list()]
            for col in columns
        ]
        return [
            np.stack([col_arrays[j][i] for j in range(k)], axis=-1)
            for i in range(n_rows)
        ]
    stacked = np.stack(col_mats, axis=-1)  # (n_rows, seq_len, k)
    return [stacked[i] for i in range(n_rows)]


def _bestof(fn, df, columns, reps=5):
    best = float("inf")
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn(df, columns)
        best = min(best, time.perf_counter() - t)
    return best, out


def main():
    columns = ("mjd", "mag", "magerr", "norm")
    for n_rows, seq_len in [(5000, 200), (20000, 100), (50000, 50)]:
        df = _make_df(n_rows, seq_len, columns)
        # warm
        _old(df, columns)
        _new(df, columns)
        _new2(df, columns)
        t_old, o_old = _bestof(_old, df, columns)
        t_new, o_new = _bestof(_new, df, columns)
        t_new2, o_new2 = _bestof(_new2, df, columns)
        identical = len(o_old) == len(o_new) and all(
            np.array_equal(a, b) for a, b in zip(o_old, o_new)
        )
        identical2 = len(o_old) == len(o_new2) and all(
            np.array_equal(a, b) for a, b in zip(o_old, o_new2)
        )
        print(
            f"n_rows={n_rows} seq_len={seq_len}: old={t_old*1e3:.1f}ms "
            f"new={t_new*1e3:.1f}ms ({t_old/t_new:.2f}x, id={identical}) "
            f"new2={t_new2*1e3:.1f}ms ({t_old/t_new2:.2f}x, id={identical2})"
        )


if __name__ == "__main__":
    main()
