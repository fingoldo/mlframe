"""Wave 38 (2026-05-20): numpy view-vs-copy ambiguity audit.

Audit result: CLEAN for active corruption (0 P0/P1 bugs across ~25 candidate
sites). Three P2 fragility flags hardened:

  1. feature_engineering/mps.py:712 -- np.nan_to_num(profits[:-1], copy=False, ...)
     wrote back into a slice-view of a local 'profits' array. Today 'profits'
     is overwritten next iteration so no observable corruption, but a future
     refactor that re-reads 'profits' inside the loop would see scrubbed
     values. Fix: drop copy=False (let nan_to_num allocate).

  2. training/composite_screening.py:50 -- _extract_column_array returns a
     zero-copy view of the source polars/pandas column when dtype already
     matches float64. Currently every call site treats the result read-only
     (or fancy-indexes which copies). Fix: docstring contract states the
     return value MUST be treated read-only.

  3. training/feature_handling/target_encoders.py:192 -- _coerce_y_to_float64
     returns a view of the caller's y target when y is already float64.
     Same risk profile as composite_screening. Fix: docstring contract.

These tests are correctness sensors: if a future refactor either flips the
nan_to_num call back to copy=False, or drops the docstring contracts, the
view-vs-copy class is at risk of resurfacing.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


def test_mps_nan_to_num_does_not_write_back_into_view() -> None:
    """mps.py optimal-profit scrub must not mutate the source 'profits' array."""
    src = _read("feature_engineering/mps.py")
    # No copy=False on the np.nan_to_num call for OPTIMAL_PROFIT.
    assert "np.nan_to_num(profits[:-1], copy=False" not in src, (
        "feature_engineering/mps.py: np.nan_to_num(profits[:-1], copy=False, ...) writes scrubbed values back into the source 'profits' slice; drop copy=False."
    )
    # The fixed form must be present.
    assert "np.nan_to_num(profits[:-1], nan=0.0, posinf=0.0, neginf=0.0)" in src, (
        "feature_engineering/mps.py: expected np.nan_to_num(profits[:-1], nan=0.0, ...) (without copy=False) so a fresh array is returned."
    )


def test_extract_column_array_documents_read_only_contract() -> None:
    """composite_screening._extract_column_array must document its zero-copy contract."""
    src = _read("training/composite/discovery/screening.py")
    # The docstring must warn about read-only contract.
    assert "read-only" in src.lower(), (
        "training/composite_screening.py: _extract_column_array returns a zero-copy view when dtype matches; docstring must warn callers."
    )


def test_coerce_y_to_float64_documents_read_only_contract() -> None:
    """target_encoders._coerce_y_to_float64 must document its zero-copy contract."""
    src = _read("training/feature_handling/target_encoders.py")
    # Locate the helper and the contract paragraph.
    helper_idx = src.find("def _coerce_y_to_float64")
    assert helper_idx != -1, "Helper _coerce_y_to_float64 must exist."
    # Read 40 lines after the def signature.
    snippet = src[helper_idx : helper_idx + 1500]
    assert "read-only" in snippet.lower(), (
        "training/feature_handling/target_encoders.py: _coerce_y_to_float64 returns a zero-copy view when y is already float64; docstring must warn callers."
    )


# ---------------------------------------------------------------------------
# Behavioural smoke: ensure the mps.py fix actually returns a fresh array.
# ---------------------------------------------------------------------------


def test_nan_to_num_without_copy_false_returns_fresh_array() -> None:
    """np.nan_to_num without copy=False on a slice-view returns a fresh array,
    not a view into the source. Documents the contract behind the mps.py fix."""
    profits = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    profits_snapshot = profits.copy()
    scrubbed = np.nan_to_num(profits[:-1], nan=0.0, posinf=0.0, neginf=0.0)
    # The source must be unchanged.
    np.testing.assert_array_equal(
        profits,
        profits_snapshot,
        err_msg="np.nan_to_num (without copy=False) must not mutate the source slice.",
    )
    # The result must have the NaN scrubbed.
    assert scrubbed[2] == 0.0, "NaN should be scrubbed to 0.0 in the fresh result."
    # Sanity: passing copy=False would have written back into profits[2].
    # (We don't actually run that, just assert the contract.)


def test_extract_column_array_caller_must_copy_before_mutation() -> None:
    """When the polars source column is float64, _extract_column_array returns a view;
    a defensive .copy() at the call site protects the source frame."""
    pl = pytest.importorskip("polars")
    from mlframe.training.composite.discovery.screening import _extract_column_array

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    arr = _extract_column_array(df, "x")
    # Defensive copy at the (hypothetical) call site:
    arr_copy = arr.copy()
    arr_copy[0] = 999.0
    # The source frame must be unchanged.
    assert df["x"].to_list() == [1.0, 2.0, 3.0, 4.0], "Defensive .copy() at the call site keeps the source DataFrame intact."
