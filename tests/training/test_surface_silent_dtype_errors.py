"""Regression: silently-swallowed dtype errors must be surfaced, not masked.

Two recurring silent-error sites in the training core:

(A) ``_train_model_with_fallback`` previously did ``return None, None`` when
    an estimator raised "pandas dtypes must be int, float or bool". That hid
    an upstream feature-typing gap and surfaced downstream as an opaque
    "model produced no predictions" failure. It must now log an ERROR (with
    the offending dtypes) and re-raise.

(B) ``_align_xgb_cat_categories`` used a bare ``except Exception`` on per-col
    ``.dtype`` access and silently skipped the column. The skip is now
    narrowed to expected exceptions and logged as a WARNING so an incomplete
    cat-alignment is visible; unexpected exceptions propagate.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from mlframe.training._training_loop import _train_model_with_fallback
from mlframe.training._eval_helpers import _align_xgb_cat_categories


class _DtypeRejectingModel:
    """Raises the exact CB/LGB unsupported-dtype message on fit."""

    def fit(self, *args, **kwargs):
        """Fit."""
        raise ValueError("pandas dtypes must be int, float or bool")


def test_dtype_error_reraises_not_silent_none(caplog):
    """Dtype error reraises not silent none."""
    model = _DtypeRejectingModel()
    df = pd.DataFrame({"a": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    target = pd.Series([0, 1])

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="pandas dtypes must be"):
            _train_model_with_fallback(
                model=model,
                model_obj=model,
                model_type_name="SomeModel",
                train_df=df,
                train_target=target,
                fit_params={},
            )
    # The actionable ERROR (with offending dtypes) must have fired -- not a
    # silent ``return None, None``.
    assert any("unsupported pandas dtype" in r.getMessage() for r in caplog.records), "expected an actionable ERROR naming the unsupported dtype"


class _DtypeRaisingSeries:
    """A column whose ``.dtype`` access raises the expected KeyError."""

    @property
    def dtype(self):
        """Dtype."""
        raise KeyError("simulated dtype-resolution failure")


class _DfWithBadCol(pd.DataFrame):
    """pandas DataFrame whose __getitem__ returns a dtype-raising series for
    a sentinel column, exercising the narrowed except in the alignment loop.
    """

    _bad_col = "bad"

    @property
    def _constructor(self):
        """Constructor."""
        return _DfWithBadCol

    def __getitem__(self, key):
        if key == self._bad_col:
            return _DtypeRaisingSeries()
        return super().__getitem__(key)


def test_align_dtype_access_failure_warns_not_silent(caplog):
    """Align dtype access failure warns not silent."""
    df = _DfWithBadCol({"bad": [1, 2], "ok": [3, 4]})

    with caplog.at_level(logging.WARNING):
        # XGB model name so alignment loop runs over the pandas frame.
        _align_xgb_cat_categories("XGBClassifier", df)

    assert any(
        "dtype access failed for col='bad'" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records
    ), "expected a WARNING surfacing the skipped column, not a silent debug"
