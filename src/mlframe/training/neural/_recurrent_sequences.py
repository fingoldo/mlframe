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

    # Convert each column's per-row list to a numpy array once. The previous
    # implementation did n_rows * n_cols Python list-lookups in a comprehension
    # then column_stack'd per row, materialising n_rows separate small (k, n_cols)
    # arrays via Python-level loops. Casting per column first lets each row's
    # stack use ndarray slicing rather than nested-list indexing.
    col_arrays: list[list[np.ndarray]] = [
        [np.asarray(v, dtype=np.float32) for v in df[col].to_list()]
        for col in columns
    ]

    result: list[np.ndarray] = [
        np.stack([col_arrays[j][i] for j in range(len(columns))], axis=-1)
        for i in range(n_rows)
    ]

    return result


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
