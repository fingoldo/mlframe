"""Unit tests for the consolidated coercion helpers in ``mlframe.training.utils``.

Replaces three near-identical ``_to_1d_numpy`` definitions previously living in
``drift_report.py``, ``baseline_diagnostics.py`` and ``composite_estimator.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from mlframe.training.utils import coerce_to_numpy, coerce_to_1d_numpy


class TestCoerceToNumpy:
    def test_pandas_series_returns_ndarray(self):
        s = pd.Series([1.0, 2.0, 3.0])
        out = coerce_to_numpy(s)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_numpy_passthrough(self):
        arr = np.array([1, 2, 3])
        out = coerce_to_numpy(arr)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, arr)

    def test_list_coerces(self):
        out = coerce_to_numpy([1, 2, 3])
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.array([1, 2, 3]))

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_polars_series_returns_ndarray(self):
        s = pl.Series("x", [1.0, 2.0, 3.0])
        out = coerce_to_numpy(s)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_none_raises_by_default(self):
        with pytest.raises(TypeError, match="allow_none"):
            coerce_to_numpy(None)

    def test_none_allowed_when_opt_in(self):
        assert coerce_to_numpy(None, allow_none=True) is None

    def test_preserves_2d_shape(self):
        """``coerce_to_numpy`` does NOT reshape (use ``coerce_to_1d_numpy`` for that)."""
        arr = np.arange(6).reshape(2, 3)
        out = coerce_to_numpy(arr)
        assert out.shape == (2, 3)


class TestCoerceTo1dNumpy:
    def test_flatten_2d(self):
        arr = np.arange(6).reshape(2, 3)
        out = coerce_to_1d_numpy(arr)
        assert out.shape == (6,)
        np.testing.assert_array_equal(out, np.arange(6))

    def test_already_1d_passthrough(self):
        out = coerce_to_1d_numpy(np.array([1.0, 2.0, 3.0]))
        assert out.shape == (3,)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_pandas_series_flattens(self):
        out = coerce_to_1d_numpy(pd.Series([1.0, 2.0, 3.0]))
        assert out.shape == (3,)

    def test_none_always_raises(self):
        """1-D variant does not have an opt-in for None (no sensible 1-D None)."""
        with pytest.raises(TypeError):
            coerce_to_1d_numpy(None)


class TestBackwardCompatAliases:
    """The three modules that used to define ``_to_1d_numpy`` locally now import
    aliases from utils. Confirm those aliases still expose the historical name."""

    def test_drift_report_alias_preserved(self):
        # drift_report uses `_to_numpy_or_none` (renamed 2026-05-15 after the
        # naming audit flagged the old `_to_1d_numpy` name as misleading -
        # the function preserves shape, doesn't reshape to 1-D).
        from mlframe.training import drift_report
        assert drift_report._to_numpy_or_none(None) is None
        np.testing.assert_array_equal(
            drift_report._to_numpy_or_none([1, 2, 3]), np.array([1, 2, 3])
        )

    def test_baseline_diagnostics_alias_preserved(self):
        from mlframe.training import baseline_diagnostics
        # baseline_diagnostics uses the 1-D variant; reshape applies
        out = baseline_diagnostics._to_1d_numpy(np.arange(4).reshape(2, 2))
        assert out.shape == (4,)

    def test_composite_estimator_alias_preserved(self):
        from mlframe.training import composite_estimator
        out = composite_estimator._to_1d_numpy(np.arange(4).reshape(2, 2))
        assert out.shape == (4,)
