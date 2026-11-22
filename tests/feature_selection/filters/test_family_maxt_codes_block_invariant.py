"""The block size of the family coding is a memory knob: the codes and counts of every column are the same whatever the block.

Pins the contract that lets the default block (sized from a 128 MB budget) and any explicit block be used interchangeably, including for a column with a non-finite value.
"""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.filters._family_maxt_gate as gate


def _matrix(seed: int = 0) -> np.ndarray:
    """A (rows, columns) float matrix with one column carrying NaN and one carrying inf."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 7))
    X[5, 2] = np.nan
    X[11, 4] = np.inf
    return X


def test_codes_are_the_same_for_the_default_block_and_every_explicit_block():
    """Coding the columns one at a time, in one block, or in blocks of two gives identical codes and counts."""
    X = _matrix()
    ref_codes, ref_nx = gate._family_codes(X, 8)
    for block in (1, 2, 3, X.shape[1]):
        codes, nx = gate._family_codes(X, 8, block=block)
        np.testing.assert_array_equal(codes, ref_codes, err_msg=f"block={block}")
        np.testing.assert_array_equal(nx, ref_nx, err_msg=f"block={block}")


def test_a_non_positive_block_is_treated_as_one_column_per_block():
    """A block of zero or less codes one column at a time instead of failing or skipping columns."""
    X = _matrix(1)
    ref_codes, ref_nx = gate._family_codes(X, 6, block=1)
    for block in (0, -4):
        codes, nx = gate._family_codes(X, 6, block=block)
        np.testing.assert_array_equal(codes, ref_codes)
        np.testing.assert_array_equal(nx, ref_nx)


def test_family_codes_of_rows_matches_family_codes_on_the_transposed_matrix():
    """The row-major entry point codes the same columns as the column-major one, with the same non-finite handling."""
    X = _matrix(2)
    codes_cols, nx_cols = gate._family_codes(X, 8)
    codes_rows, nx_rows = gate.family_codes_of_rows(np.ascontiguousarray(X.T), 8)
    np.testing.assert_array_equal(codes_rows, codes_cols)
    np.testing.assert_array_equal(nx_rows, nx_cols)
