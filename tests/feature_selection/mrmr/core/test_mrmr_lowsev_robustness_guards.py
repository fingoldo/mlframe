"""Robustness guards for degenerate inputs in the MRMR discretisation / artifact / validation path.

Each test pins a guard that turns an opaque crash (empty-axis reduction, negative-code ``np.bincount``, object-dtype inf
slipping past the float-only scan) into clean handling or a clear, source-naming error. The constant-y / inf validation
scans additionally log (do not re-raise) any unexpected failure so a swallowed bug stays traceable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization._discretization_dataset import categorize_dataset
from mlframe.feature_selection.filters._mrmr_artifacts import _assert_nonneg_codes
from mlframe.feature_selection.filters.mrmr import MRMR
import mlframe.feature_selection.filters._mrmr_validate_transform as VT


# --------------------------------------------------------------------------
# B11: categorize_dataset must not crash on a frame that yields zero usable
# columns OR an empty reduction axis (the final ``data.max(axis=0)`` and the
# in-loop categorical ``new_vals.max(axis=0)`` both raise on an empty axis).
# --------------------------------------------------------------------------


@pytest.mark.fast
def test_categorize_dataset_zero_row_categorical_returns_empty_not_crash():
    """A 0-row categorical-only frame reaches both ``max(axis=0)`` sites with an empty reduction axis.

    Pre-fix: ``ValueError: zero-size array to reduction operation maximum which has no identity``.
    Post-fix: a typed ``(data, cols, nbins)`` triple with zero-valued nbins and no crash.
    """
    df = pd.DataFrame({"a": pd.Series([], dtype="object"), "b": pd.Series([], dtype="object")})
    data, cols, nbins = categorize_dataset(df)
    assert data.shape == (0, 2)
    assert cols == ["a", "b"]
    assert nbins.tolist() == [0, 0]


@pytest.mark.fast
def test_categorize_dataset_zero_column_frame_returns_empty():
    """A frame with rows but zero columns returns an empty result without crashing."""
    df = pd.DataFrame(index=range(5))
    data, cols, nbins = categorize_dataset(df)
    assert data.shape[1] == 0
    assert cols == []
    assert nbins.tolist() == []


@pytest.mark.fast
def test_categorize_dataset_normal_frame_unaffected():
    """The guard does not perturb the normal (non-degenerate) path."""
    df = pd.DataFrame({"x": np.arange(50, dtype=float), "y": np.arange(50, dtype=float)[::-1]})
    data, cols, nbins = categorize_dataset(df, n_bins=4)
    assert data.shape == (50, 2)
    assert cols == ["x", "y"]
    assert (nbins > 1).all()


# --------------------------------------------------------------------------
# B12: ``_assert_nonneg_codes`` fails loudly (naming the cause) on a negative
# bin code, instead of letting ``np.bincount`` raise its opaque error.
# --------------------------------------------------------------------------


@pytest.mark.fast
def test_assert_nonneg_codes_raises_clear_error_on_negative():
    """Assert nonneg codes raises clear error on negative."""
    codes = np.array([0, 1, -1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="negative bin code"):
        _assert_nonneg_codes(codes, "feature column 'foo' x_bins")


@pytest.mark.fast
def test_assert_nonneg_codes_passthrough_on_valid_and_empty():
    """Assert nonneg codes passthrough on valid and empty."""
    _assert_nonneg_codes(np.array([0, 1, 2, 3], dtype=np.int64), "ok")  # no raise
    _assert_nonneg_codes(np.array([], dtype=np.int64), "empty")  # no raise


@pytest.mark.fast
def test_compute_mrmr_artifacts_guard_triggers_on_negative_target_code():
    """The guard fires from inside ``compute_mrmr_artifacts`` when a target column carries a negative code."""
    from mlframe.feature_selection.filters._mrmr_artifacts import compute_mrmr_artifacts

    n = 8
    data = np.zeros((n, 2), dtype=np.int64)
    data[:, 0] = np.array([0, 1, 0, 1, -1, 0, 1, 0])  # deliberate negative in the target column
    nbins = np.array([2, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="negative bin code"):
        compute_mrmr_artifacts(
            data=data,
            cols=["y", "f0"],
            nbins=nbins,
            target_indices=np.array([0]),
            cached_MIs={},
            feature_names_in=["f0"],
            support_original=np.array([1]),
            retain_bins=False,
            dtype=np.int64,
        )


# --------------------------------------------------------------------------
# B14: object-dtype column smuggling a Python float('inf') must be caught by
# MRMR._validate_inputs, not slip through to the discretiser.
# --------------------------------------------------------------------------


@pytest.mark.fast
def test_validate_inputs_catches_inf_in_object_column():
    """Validate inputs catches inf in object column."""
    m = MRMR()
    y = np.array([0, 1, 0, 1, 0, 1])
    df = pd.DataFrame({"a": pd.Series([1.0, 2.0, float("inf"), 4.0, 5.0, 6.0], dtype="object"), "b": [1, 2, 3, 4, 5, 6]})
    with pytest.raises(ValueError, match="object-dtype column"):
        m._validate_inputs(df, y)


@pytest.mark.fast
def test_validate_inputs_clean_object_column_no_false_positive():
    """Validate inputs clean object column no false positive."""
    m = MRMR()
    y = np.array([0, 1, 0, 1, 0, 1])
    df = pd.DataFrame({"a": pd.Series(["x", "y", "z", "w", "u", "v"], dtype="object"), "b": [1, 2, 3, 4, 5, 6]})
    m._validate_inputs(df, y)  # no raise


@pytest.mark.fast
def test_validate_inputs_float_inf_still_caught():
    """Validate inputs float inf still caught."""
    m = MRMR()
    y = np.array([0, 1, 0, 1, 0, 1])
    df = pd.DataFrame({"a": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    with pytest.raises(ValueError, match=r"contains \+/-inf"):
        m._validate_inputs(df, y)


# --------------------------------------------------------------------------
# F6: the inf / constant-y validation scans log (with traceback) any
# unexpected non-ValueError failure rather than swallowing it silently, and
# control flow continues unchanged.
# --------------------------------------------------------------------------


@pytest.mark.fast
def test_constant_y_scan_logs_unexpected_failure(monkeypatch, caplog):
    """Constant y scan logs unexpected failure."""
    m = MRMR()
    y = np.array([0, 1, 0, 1, 0, 1])
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    def _boom(*a, **k):
        """Helper that boom."""
        raise RuntimeError("injected scan failure")

    monkeypatch.setattr(VT.np, "unique", _boom)
    with caplog.at_level(logging.DEBUG, logger=VT.logger.name):
        m._validate_inputs(df, y)  # swallowed + logged, no raise
    assert any("validation scan failed" in r.message for r in caplog.records)
    assert any(r.exc_info is not None for r in caplog.records)
