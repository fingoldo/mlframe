"""Integer-lattice & pairwise-modular FE replay must NOT emit INT64_MIN garbage for non-finite operands
(target-leakage / replay audit, 2026-06-13).

Both generators ship default-ON and claim "pure integer arithmetic on X, train/test bit-identical". The
eligibility scan guarantees finite integer-valued operands at FIT, but the apply path cast operands to
int64 with no re-check: a drifted/test frame carrying NaN or inf in a source column that was
integer-valued on train casts to INT64_MIN (np.asarray(nan).astype(int64) == -9223372036854775808), which
gcd/lcm/bitwise_and/mod turn into a WRONG, NON-NaN value -- silent garbage indistinguishable downstream
from a genuine feature value. The fix NaN-outs rows where any operand is non-finite (the feature is
undefined there) while leaving clean rows byte-identical.

These tests pin: (1) a non-finite operand -> NaN output, never the INT64_MIN-derived garbage, and (2)
clean rows are byte-identical whether or not other rows are non-finite (the mask does not perturb them).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._integer_lattice_fe import (
    apply_integer_lattice,
    _lattice_columns_for_pair,
)
from mlframe.feature_selection.filters._pairwise_modular_fe import apply_pairwise_modular

_INT64_MIN = np.int64(np.iinfo(np.int64).min)


class _DictX:
    """Minimal column-addressable X (apply_* index columns by name)."""

    def __init__(self, cols):
        self._c = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}

    def __getitem__(self, k):
        return self._c[k]


@pytest.mark.parametrize("op", ["gcd", "lcm", "bitwise_and"])
def test_integer_lattice_nonfinite_operand_is_nan_not_garbage(op):
    """Integer lattice nonfinite operand is nan not garbage."""
    a = np.array([12.0, 18.0, np.nan, 24.0, np.inf], dtype=np.float64)
    b = np.array([8.0, 6.0, 4.0, np.nan, 2.0], dtype=np.float64)
    out = apply_integer_lattice(_DictX({"a": a, "b": b}), op, ("a", "b"))
    # rows 2,3,4 have a non-finite operand -> must be NaN (undefined), NEVER an INT64_MIN-derived integer.
    assert np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[4]), f"{op}: non-finite operand leaked a value: {out}"
    # the garbage the bug produced (gcd/and/mod of INT64_MIN) would be a huge finite number -- assert none survive.
    assert not np.any(np.isfinite(out) & (np.abs(out) > 1e15)), f"{op}: INT64_MIN-scale garbage present: {out}"
    # clean rows (0,1) keep their genuine values.
    assert np.isfinite(out[0]) and np.isfinite(out[1])


def test_integer_lattice_clean_rows_byte_identical_with_and_without_nan_neighbours():
    """Integer lattice clean rows byte identical with and without nan neighbours."""
    a_clean = np.array([12.0, 18.0, 24.0, 30.0], dtype=np.float64)
    b_clean = np.array([8.0, 6.0, 16.0, 10.0], dtype=np.float64)
    ref = apply_integer_lattice(_DictX({"a": a_clean, "b": b_clean}), "gcd", ("a", "b"))
    # inject a NaN row in the middle; the clean rows must be byte-identical to the all-clean run.
    a_drift = np.array([12.0, 18.0, np.nan, 24.0, 30.0], dtype=np.float64)
    b_drift = np.array([8.0, 6.0, 4.0, 16.0, 10.0], dtype=np.float64)
    drift = apply_integer_lattice(_DictX({"a": a_drift, "b": b_drift}), "gcd", ("a", "b"))
    keep = np.array([0, 1, 3, 4])  # the clean rows in the drifted frame
    assert np.array_equal(drift[keep], ref), "clean rows perturbed by a non-finite neighbour (mask must be row-local)"
    assert np.isnan(drift[2])


def test_lattice_batched_matches_and_masks():
    # the per-pair batched path (_lattice_columns_for_pair) must mask the same rows as the per-column path.
    """Lattice batched matches and masks."""
    a = np.array([12.0, np.nan, 24.0], dtype=np.float64)
    b = np.array([8.0, 6.0, 16.0], dtype=np.float64)
    mat = _lattice_columns_for_pair(a, b, ("gcd", "lcm", "bitwise_and"))
    assert np.all(np.isnan(mat[1, :])), "batched path left INT64_MIN garbage on the non-finite row"
    assert np.all(np.isfinite(mat[[0, 2], :])), "batched path NaN'd clean rows"


@pytest.mark.parametrize("op", ["sum", "diff", "prod"])
def test_pairwise_modular_nonfinite_operand_is_nan_not_garbage(op):
    """Pairwise modular nonfinite operand is nan not garbage."""
    a = np.array([10.0, 21.0, np.nan, 33.0], dtype=np.float64)
    b = np.array([3.0, 7.0, 5.0, np.inf], dtype=np.float64)
    out = apply_pairwise_modular(_DictX({"a": a, "b": b}), op, ("a", "b"), modulus=7)
    assert np.isnan(out[2]) and np.isnan(out[3]), f"{op}: non-finite operand produced a residue: {out}"
    # genuine residues live in [0, 7); the INT64_MIN bug produced an in-range-but-wrong residue, so the
    # discriminating check is the NaN above. Clean rows must be valid residues.
    clean = out[[0, 1]]
    assert np.all((clean >= 0) & (clean < 7)), f"clean rows not valid residues: {out}"


def test_pairwise_modular_clean_rows_byte_identical_with_nan_neighbour():
    """Pairwise modular clean rows byte identical with nan neighbour."""
    a_clean = np.array([10.0, 21.0, 33.0], dtype=np.float64)
    b_clean = np.array([3.0, 7.0, 4.0], dtype=np.float64)
    ref = apply_pairwise_modular(_DictX({"a": a_clean, "b": b_clean}), "sum", ("a", "b"), modulus=5)
    a_drift = np.array([10.0, 21.0, np.nan, 33.0], dtype=np.float64)
    b_drift = np.array([3.0, 7.0, 9.0, 4.0], dtype=np.float64)
    drift = apply_pairwise_modular(_DictX({"a": a_drift, "b": b_drift}), "sum", ("a", "b"), modulus=5)
    keep = np.array([0, 1, 3])
    assert np.array_equal(drift[keep], ref), "modular clean rows perturbed by a non-finite neighbour"
    assert np.isnan(drift[2])
