"""Identity-pin for the extract_sequences bulk-stack fast path.

extract_sequences has an equal-length fast path (one np.asarray per column over
the whole list-of-lists + a single np.stack) with a per-row fallback for ragged
sequence lengths. Both paths must yield bit-identical float32 (seq_len, n_cols)
arrays in column order. These tests pin both branches.
"""

from __future__ import annotations

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from mlframe.training.neural import extract_sequences


COLUMNS = ("mjd", "mag", "magerr", "norm")


def _reference(df, columns):
    """The pre-optimization per-row stack, kept here as the identity oracle."""
    n_rows = len(df)
    col_arrays = [[np.asarray(v, dtype=np.float32) for v in df[col].to_list()] for col in columns]
    return [np.stack([col_arrays[j][i] for j in range(len(columns))], axis=-1) for i in range(n_rows)]


def test_equal_length_fast_path_bit_identical():
    rng = np.random.default_rng(0)
    n_rows, seq_len = 200, 30
    data = {c: [rng.standard_normal(seq_len).tolist() for _ in range(n_rows)] for c in COLUMNS}
    df = pl.DataFrame(data)
    out = extract_sequences(df, columns=COLUMNS)
    ref = _reference(df, COLUMNS)
    assert len(out) == len(ref) == n_rows
    for a, b in zip(out, ref):
        assert a.dtype == np.float32
        assert a.shape == (seq_len, len(COLUMNS))
        assert np.array_equal(a, b)


def test_ragged_fallback_bit_identical():
    rng = np.random.default_rng(1)
    n_rows = 150
    lens = [int(rng.integers(2, 12)) for _ in range(n_rows)]
    data = {c: [rng.standard_normal(lens[i]).tolist() for i in range(n_rows)] for c in COLUMNS}
    df = pl.DataFrame(data)
    out = extract_sequences(df, columns=COLUMNS)
    ref = _reference(df, COLUMNS)
    assert len(out) == len(ref) == n_rows
    for a, b, ln in zip(out, ref, lens):
        assert a.dtype == np.float32
        assert a.shape == (ln, len(COLUMNS))
        assert np.array_equal(a, b)


def test_indices_subset_and_column_order():
    rng = np.random.default_rng(2)
    n_rows, seq_len = 80, 10
    cols = ("a", "b")
    data = {c: [rng.standard_normal(seq_len).tolist() for _ in range(n_rows)] for c in cols}
    df = pl.DataFrame(data)
    idx = [5, 1, 70, 33]
    out = extract_sequences(df, indices=idx, columns=cols)
    ref = _reference(df[idx], cols)
    assert len(out) == len(idx)
    for a, b in zip(out, ref):
        assert np.array_equal(a, b)
