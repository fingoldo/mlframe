"""RFECV does not build a float64 twin of a pandas X above the size gate (tracker 2026-11-02 wave 16, MEM16-3)."""

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.wrappers.rfecv import _fit


def _frame():
    """A small all-numeric float32 frame."""
    return pd.DataFrame(np.random.default_rng(0).normal(size=(100, 5)).astype(np.float32), columns=list("abcde"))


def test_small_frame_still_gets_the_float64_view():
    """Below the gate the conversion is unchanged: a C-contiguous float64 array and a column map."""
    arr, pos = _fit._numpy_view_of_numeric_X(False, _frame(), None, None)
    assert arr is not None and arr.dtype == np.float64 and pos["c"] == 2


def test_frame_above_the_gate_keeps_the_pandas_path_and_logs(monkeypatch, caplog):
    """With a gate below the frame's float64 size no copy is made and the choice is logged."""
    monkeypatch.setattr(_fit, "_NUMPY_VIEW_MAX_BYTES", 100)
    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.wrappers.rfecv"):
        arr, pos = _fit._numpy_view_of_numeric_X(False, _frame(), None, None)
    assert arr is None and pos is None
    assert any("float64 numpy view" in r.getMessage() for r in caplog.records)
