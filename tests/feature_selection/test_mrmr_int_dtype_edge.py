"""Edge cases for the INTEGER / codes-matrix type handling: compact-codes downcast boundary, the categorical
cardinality cap (int8 boundary, NaN sentinel, degenerate columns), sentinel consistency (no ``-1`` reaches the
joint kernels), and the joint dtype staying wide under a narrow storage matrix. Companion to
``test_mrmr_compact_codes.py`` / ``test_categorical_cardinality_cap.py``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization import (
    cap_categorical_cardinality,
    categorize_dataset,
)
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars


# --------------------------------------------------------------------------------------------------
# cap_categorical_cardinality: int8 boundary + NaN sentinel + degenerate columns
# --------------------------------------------------------------------------------------------------


def test_cap_127_with_nan_sentinel_fits_int8_after_shift():
    """cap=127 must yield max code 126 (other bucket), so the later ``+1`` NaN shift (categorize_dataset) lands at
    127 -- exactly the int8 ceiling, no overflow. The ``-1`` sentinel is preserved (shift converts it to bin 0)."""
    col = np.repeat(np.arange(200.0), 20).reshape(-1, 1).copy()
    col[:50] = -1.0
    capped = cap_categorical_cardinality(col, 127)
    assert int(capped.max()) == 126, "cap=127 -> other bucket is code 126"
    assert (capped == -1).any(), "NaN sentinel preserved by the cap"
    # Emulate the categorize_dataset +1 NaN shift; result must fit int8 (max value 127).
    shifted = capped + 1
    assert shifted.max() <= 127, "after the NaN +1 shift the max code is 127, still int8"


def test_cap_boundary_128_and_129_distinct():
    """A column with 128 distinct dense codes (0..127) already fits int8; capping at 127 folds it to 0..126. A 300-card
    column capped at 128 lands at 127 (max int8), capped at 129 lands at 128 (needs int16)."""
    c300 = np.repeat(np.arange(300.0), 5).reshape(-1, 1)
    assert int(cap_categorical_cardinality(c300, 128).max()) == 127
    assert int(cap_categorical_cardinality(c300, 129).max()) == 128  # 128 > int8 -> downcast must pick int16


def test_cap_handles_non_dense_codes_with_unused_gap():
    """cap keys off ``finite.max()+1`` (assumes dense factorize codes); a non-dense column with a gap (unused category)
    must still remap correctly -- bincount(minlength) zero-fills the gap, argsort folds it into 'other'."""
    gap = np.array([[0.0], [2.0], [0.0], [2.0], [2.0]])  # codes {0,2}, category 1 unused
    out = cap_categorical_cardinality(gap, 2).ravel()
    # 2 is the most frequent -> kept as 0; 0 folds into the other bucket 1. Dense output, no crash on the gap.
    assert set(np.unique(out)) == {0.0, 1.0}
    assert (out == np.array([1.0, 0.0, 1.0, 0.0, 0.0])).all()


def test_cap_degenerate_columns():
    """Single-category, all-NaN, empty, and cap=2 columns must not crash and stay in-range."""
    assert (cap_categorical_cardinality(np.zeros((5, 1)), 2).ravel() == 0).all(), "single category -> code 0"
    allnan = np.full((5, 1), -1.0)
    assert (cap_categorical_cardinality(allnan, 3) == -1).all(), "all-NaN column unchanged (all sentinel)"
    assert cap_categorical_cardinality(np.empty((0, 3)), 5).size == 0, "empty frame -> empty"
    two = np.array([[0.0], [1.0], [0.0], [1.0], [2.0]])
    assert int(cap_categorical_cardinality(two, 2).max()) == 1, "cap=2 -> max code 1"


# --------------------------------------------------------------------------------------------------
# Compact-codes downcast boundary (the range-checked int8/int16 selection used in _fit_impl_core)
# --------------------------------------------------------------------------------------------------


def _downcast(data: np.ndarray):
    """Replicates the COMPACT CODES STORAGE range check in _fit_impl_core."""
    dmin, dmax = int(data.min()), int(data.max())
    if -128 <= dmin and dmax <= 127:
        return np.int8
    if -32768 <= dmin and dmax <= 32767:
        return np.int16
    return None


def test_downcast_boundary_is_exact_at_127():
    """max code exactly 127 -> int8; 128 -> int16 (off-by-one guard on the range check)."""
    d127 = np.arange(128, dtype=np.int32).reshape(-1, 1)  # 0..127
    assert _downcast(d127) is np.int8
    assert np.array_equal(d127.astype(np.int8).astype(np.int64), d127.astype(np.int64))
    d128 = np.arange(129, dtype=np.int32).reshape(-1, 1)  # 0..128
    assert _downcast(d128) is np.int16, "code 128 overflows int8 -> must widen to int16"


# --------------------------------------------------------------------------------------------------
# Sentinel consistency: the categorize_dataset codes matrix never contains a raw -1 (the joint kernels
# index freqs by the raw code; a -1 would wrap to the last bin). Categorical NaN is +1 shifted.
# --------------------------------------------------------------------------------------------------


def test_categorical_nan_shifted_no_negative_codes_reach_matrix():
    """A categorical column with NaN must emerge with codes >= 0 (NaN -> its own bin 0), never the raw ``-1`` sentinel
    that pd.factorize / .cat.codes emit -- otherwise the joint-histogram kernels index out of range / wrap."""
    n = 500
    cat = np.array(["a", "b", "c"] * (n // 3) + ["a", "b"])[:n].astype(object)
    cat[::30] = None  # inject NaN
    X = pd.DataFrame({"num": np.random.default_rng(0).normal(size=n), "cat": pd.Categorical(cat)})
    data, cols, nbins = categorize_dataset(df=X, missing_strategy="separate_bin", dtype=np.int16)
    assert data.min() >= 0, "no raw -1 sentinel may reach the codes matrix (NaN is +1 shifted to bin 0)"
    ci = cols.index("cat")
    assert data[:, ci].max() < nbins[ci], "every code is a valid index into its column's nbins histogram"


# --------------------------------------------------------------------------------------------------
# Joint dtype stays wide under a narrow storage matrix (int8 factors_data -> int32/int64 joint math).
# --------------------------------------------------------------------------------------------------


def test_merge_vars_joint_bit_identical_across_storage_dtype():
    """merge_vars casts the narrow storage column UP to its ``dtype`` and counts joints in int64, so an int8-stored
    matrix yields BIT-IDENTICAL joint frequencies to the same matrix stored int32 -- the basis of compact-codes
    selection-equivalence."""
    rng = np.random.default_rng(0)
    n, p = 4000, 3
    mat = rng.integers(0, 12, size=(n, p)).astype(np.int32)
    nbins = np.full(p, 12, dtype=np.int64)
    idx = np.array([0, 1, 2], dtype=np.int64)
    _, freqs_wide, nc_wide = merge_vars(mat, idx, None, nbins, dtype=np.int32)
    _, freqs_narrow, nc_narrow = merge_vars(mat.astype(np.int8), idx, None, nbins, dtype=np.int32)
    assert nc_wide == nc_narrow
    assert np.array_equal(freqs_wide, freqs_narrow), "int8 storage must give identical joint freqs to int32 storage"


def test_merge_vars_joint_counter_survives_large_single_bin():
    """The joint bin counter is int64 regardless of the narrow storage / caller dtype: a single bin holding > 127
    samples must NOT overflow (historically an int8 counter wrapped a 200-sample bin negative)."""
    n = 5000
    mat = np.zeros((n, 1), dtype=np.int8)  # all rows in bin 0 -> one bin with 5000 counts
    nbins = np.array([2], dtype=np.int64)
    _, freqs, _ = merge_vars(mat, np.array([0], dtype=np.int64), None, nbins, dtype=np.int8)
    assert np.isclose(freqs.sum(), 1.0), "normalized freqs must sum to 1 (no negative-wrapped counter)"
    assert (freqs >= 0).all(), "no bin frequency may be negative (int64 counter, not int8)"


# --------------------------------------------------------------------------------------------------
# Polars regression: max_categorical_cardinality was silently IGNORED on polars input (only the pandas
# branch applied the cap). Fails pre-fix (max code == raw cardinality-1).
# --------------------------------------------------------------------------------------------------


def test_polars_input_honors_categorical_cardinality_cap():
    pl = pytest.importorskip("polars")
    n = 6000
    df = pd.DataFrame({
        "num": np.random.default_rng(0).normal(size=n),
        "hc": pd.Categorical(np.repeat(np.arange(300), 20)[:n].astype(str)),
    })
    pdf = pl.from_pandas(df)
    data, cols, nbins = categorize_dataset(df=pdf, max_categorical_cardinality=10, dtype=np.int32)
    hc = cols.index("hc")
    assert int(data[:, hc].max()) <= 9, "polars path must fold the rare tail to fit cap=10 (was 299 pre-fix)"
    assert int(nbins[hc]) <= 10
    # Sanity: without the cap the polars path keeps the full cardinality.
    d0, c0, _ = categorize_dataset(df=pdf, dtype=np.int32)
    assert int(d0[:, c0.index("hc")].max()) == 299, "no-cap polars path keeps full cardinality"
