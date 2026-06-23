"""Sequence-extraction utilities carved out of ``recurrent.py`` to keep the
parent facade under the 1000-LOC budget. ``recurrent.py`` re-exports both
names from its bottom so existing imports keep resolving unchanged.
"""
from __future__ import annotations

import numpy as np
import polars.dataframe as pl_df


def extract_sequences(
    df: pl_df.DataFrame,
    indices: np.ndarray | list[int] | None = None,
    columns: tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> list[np.ndarray]:
    """
    Extract raw time series from Polars DataFrame with list columns.

    Args:
        df: DataFrame with list columns
        indices: Optional subset of row indices to extract
        columns: Column names to stack into sequences

    Returns:
        List of (seq_len, n_columns) float32 arrays
    """
    if indices is not None:
        df = df[indices]

    n_rows = len(df)

    # Equal-length fast path (the aligned light-curve layout: every column's per-row
    # list has the same length within a row): convert each column's whole list-of-lists
    # to one (n_rows, seq_len) ndarray in a single np.asarray, stack the columns once
    # into (n_rows, seq_len, k), and slice out the rows. This avoids the per-cell
    # np.asarray (n_rows * n_cols Python calls) and the n_rows separate np.stack calls
    # the row-wise path makes. Ragged input (unequal per-row lengths) raises on the
    # bulk np.asarray; we catch it and fall back to the exact per-row stack -- the two
    # paths are bit-identical (same float32 cast, same column order).
    n_cols = len(columns)
    try:
        col_mats = [np.asarray(df[col].to_list(), dtype=np.float32) for col in columns]
        if any(m.ndim != 2 for m in col_mats):
            raise ValueError("ragged sequence lengths")
        stacked = np.stack(col_mats, axis=-1)  # (n_rows, seq_len, k)
        return [stacked[i] for i in range(n_rows)]
    except ValueError:
        col_arrays: list[list[np.ndarray]] = [
            [np.asarray(v, dtype=np.float32) for v in df[col].to_list()]
            for col in columns
        ]
        return [
            np.stack([col_arrays[j][i] for j in range(n_cols)], axis=-1)
            for i in range(n_rows)
        ]


def extract_sequences_chunked(
    df: pl_df.DataFrame,
    indices: np.ndarray | list[int] | None = None,
    chunk_size: int = 100_000,
    columns: tuple[str, ...] = ("mjd", "mag", "magerr", "norm"),
) -> list[np.ndarray]:
    """
    Memory-efficient sequence extraction for large datasets.

    Args:
        df: DataFrame with list columns
        indices: Optional subset of row indices
        chunk_size: Number of rows per chunk
        columns: Column names to extract

    Returns:
        List of (seq_len, n_columns) float32 arrays
    """
    if indices is not None:
        indices = np.asarray(indices)
    else:
        indices = np.arange(len(df))

    sequences: list[np.ndarray] = []

    for start in range(0, len(indices), chunk_size):
        chunk_indices = indices[start : start + chunk_size]
        chunk_seqs = extract_sequences(df, chunk_indices.tolist(), columns)
        sequences.extend(chunk_seqs)

    return sequences
