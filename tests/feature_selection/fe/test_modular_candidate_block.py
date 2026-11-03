"""The row-matrix modular candidate pool equals the generator's DataFrame, and its family codes equal the DataFrame path's."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._family_maxt_gate import _family_codes, family_codes_of_rows
from mlframe.feature_selection.filters._periodic_fe import generate_modular_features, modular_candidate_block


def _frame(n: int = 4000) -> pd.DataFrame:
    """Numeric columns of several shapes, one with non-finite values."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(n) * 50
    a[::97] = np.nan
    return pd.DataFrame({"a": a, "b": rng.integers(0, 400, n).astype(float), "c": rng.exponential(30, n), "d": rng.integers(0, 2, n)})


def test_block_rows_equal_the_generator_columns():
    """Same names in the same order, same values row by row."""
    X = _frame()
    periods = (7, 12, 24.5, 365)
    eng = generate_modular_features(X, num_cols=list(X.columns), periods=periods)
    names, values = modular_candidate_block(X, list(X.columns), periods)
    assert names == [str(c) for c in eng.columns]
    assert values.flags.c_contiguous and values.shape == (eng.shape[1], len(X))
    assert np.array_equal(values, eng.to_numpy().T)


def test_row_codes_equal_the_dataframe_codes():
    """``family_codes_of_rows`` gives the codes and code counts of ``_family_codes`` on the DataFrame, including a row with non-finite values."""
    X = _frame()
    eng = generate_modular_features(X, num_cols=list(X.columns), periods=(7, 30))
    values = np.array(eng.to_numpy().T, order="C", copy=True)
    values[1, ::11] = np.nan  # the generator scrubs, a caller may not: the median fix must match
    eng.iloc[::11, 1] = np.nan
    want_codes, want_nx = _family_codes(eng, 10)
    got_codes, got_nx = family_codes_of_rows(values, 10)
    assert np.array_equal(got_codes, want_codes)
    assert np.array_equal(got_nx, want_nx)
