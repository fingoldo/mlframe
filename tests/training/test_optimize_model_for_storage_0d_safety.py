"""Regression test for the 0-d ndarray crash in optimize_model_for_storage.

Pre-fix at train_eval.py:222 the code did
    model_columns = list(model.columns) if not isinstance(model.columns, list) else model.columns
which raises ``TypeError: iteration over a 0-d array`` whenever
``model.columns`` is a 0-d numpy scalar (single-column edge case). This
surfaced on c0030_beb1dc9b @200k regression where the MLP path
constructed model.columns as a 0-d scalar; the suite aborted 376s into
the fit, losing all prior work in that suite call.

The fix preserves the list / pd.Index / 1-d-ndarray paths unchanged and
adds an explicit ``np.ndarray AND ndim == 0`` branch that unwraps via
``.item()`` before the equality check.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_optimize_for_storage_0d_columns_match_sets_none():
    """0-d ndarray columns equal to single-element metadata -> columns = None."""
    from mlframe.training.train_eval import optimize_model_for_storage
    from mlframe.training.configs import TargetTypes

    m = SimpleNamespace()
    m.columns = np.array("only_col")  # 0-d str ndarray
    assert m.columns.ndim == 0
    optimize_model_for_storage(m, TargetTypes.REGRESSION, metadata_columns=["only_col"])
    assert m.columns is None


def test_optimize_for_storage_0d_columns_no_match_preserved():
    """0-d ndarray columns NOT matching metadata -> columns left alone (no crash)."""
    from mlframe.training.train_eval import optimize_model_for_storage
    from mlframe.training.configs import TargetTypes

    m = SimpleNamespace()
    m.columns = np.array("only_col")
    optimize_model_for_storage(m, TargetTypes.REGRESSION, metadata_columns=["other_col"])
    assert isinstance(m.columns, np.ndarray)
    assert m.columns.ndim == 0
    assert str(m.columns) == "only_col"


def test_optimize_for_storage_1d_ndarray_columns_match_sets_none():
    """1-d ndarray (the common case) still works after the fix."""
    from mlframe.training.train_eval import optimize_model_for_storage
    from mlframe.training.configs import TargetTypes

    m = SimpleNamespace()
    m.columns = np.array(["a", "b", "c"])
    optimize_model_for_storage(m, TargetTypes.REGRESSION, metadata_columns=["a", "b", "c"])
    assert m.columns is None


def test_optimize_for_storage_list_columns_match_sets_none():
    """List-of-str columns (most common): unchanged behaviour."""
    from mlframe.training.train_eval import optimize_model_for_storage
    from mlframe.training.configs import TargetTypes

    m = SimpleNamespace()
    m.columns = ["a", "b", "c"]
    optimize_model_for_storage(m, TargetTypes.REGRESSION, metadata_columns=["a", "b", "c"])
    assert m.columns is None


def test_optimize_for_storage_pandas_index_columns_match_sets_none():
    """pd.Index columns: unchanged behaviour (already handled by list(...))."""
    from mlframe.training.train_eval import optimize_model_for_storage
    from mlframe.training.configs import TargetTypes

    m = SimpleNamespace()
    m.columns = pd.Index(["x", "y"])
    optimize_model_for_storage(m, TargetTypes.REGRESSION, metadata_columns=["x", "y"])
    assert m.columns is None


def test_optimize_for_storage_metadata_columns_none_no_op():
    """When metadata_columns is None, the columns attr is left untouched
    regardless of its shape -- the cleanup path is gated."""
    from mlframe.training.train_eval import optimize_model_for_storage
    from mlframe.training.configs import TargetTypes

    m = SimpleNamespace()
    m.columns = np.array("only_col")  # 0-d ndarray that would crash list()
    optimize_model_for_storage(m, TargetTypes.REGRESSION, metadata_columns=None)
    # No crash + columns preserved.
    assert m.columns is not None
