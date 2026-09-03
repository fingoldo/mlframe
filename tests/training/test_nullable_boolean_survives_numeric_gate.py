"""A nullable boolean column must reach the sklearn-bridge pipeline, not be dropped as "non-numeric".

pandas' ``bool`` dtype cannot hold NA, so a polars Boolean with any null converts to ``object``. That made it
invisible to the bool-to-int8 promotion and then to the numeric gate, which dropped it and told the operator to
"encode these upstream" -- advice that makes no sense for a column that was already boolean.

Production case: ``hide_budget`` with 476,193 False, 6 True and 2,217,431 null. Most of its signal is in the
missingness, so the null has to survive the promotion too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.pipeline._pipeline_extensions import _boolish_object_columns, _filter_to_numeric


def _frame(with_nulls: bool = True) -> pd.DataFrame:
    """A polars frame converted the way the suite converts it, with an optionally-nullable boolean."""
    flags = [True, False, None, False, True] if with_nulls else [True, False, False, False, True]
    return pl.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0], "flag": flags, "txt": ["a", "b", "c", "d", "e"]}).to_pandas()


class TestTheDtypeTrap:
    """Pins the mechanism, so a future pandas/polars version changing it is caught here rather than in a log."""

    def test_a_nullable_polars_boolean_becomes_object_in_pandas(self):
        """The whole cause in one assertion."""
        df = _frame(with_nulls=True)
        assert df["flag"].dtype == object
        assert list(df.select_dtypes(include="bool").columns) == []
        assert list(df.select_dtypes(include="number").columns) == ["num"]

    def test_a_non_null_boolean_stays_a_real_bool(self):
        """The contrast: without nulls the old code path worked, which is why this went unnoticed."""
        df = _frame(with_nulls=False)
        assert df["flag"].dtype == bool

    def test_boolish_detector_finds_it(self):
        """Detection is by VALUE, since the dtype has already lost the information."""
        assert _boolish_object_columns(_frame(with_nulls=True)) == ["flag"]

    def test_boolish_detector_ignores_real_object_columns(self):
        """A text column must not be mistaken for a boolean and cast to float."""
        assert "txt" not in _boolish_object_columns(_frame(with_nulls=True))

    def test_all_null_object_column_is_not_claimed(self):
        """Nothing to infer from an empty column; leave it to the existing all-null handling."""
        df = pd.DataFrame({"empty": pd.Series([None, None], dtype=object)})
        assert _boolish_object_columns(df) == []


class TestItSurvivesTheGate:
    """What the fix is actually for."""

    def test_nullable_boolean_is_kept(self):
        """The defect: this column used to be dropped with a warning telling the operator to encode it."""
        kept, dropped = _filter_to_numeric(_frame(with_nulls=True))
        assert "flag" in kept.columns
        assert "flag" not in dropped
        assert dropped == ["txt"]

    def test_the_nulls_survive_as_nan(self):
        """The missingness IS the signal on a column that is 82% null, so it must not become a filled 0."""
        kept, _ = _filter_to_numeric(_frame(with_nulls=True))
        values = kept["flag"].to_numpy()
        assert values[0] == 1.0 and values[1] == 0.0
        assert np.isnan(values[2])

    def test_promoted_column_is_numeric_for_sklearn(self):
        """The gate exists because the downstream transforms reject object dtype."""
        kept, _ = _filter_to_numeric(_frame(with_nulls=True))
        assert pd.api.types.is_numeric_dtype(kept["flag"])

    def test_val_and_test_follow_the_train_schema(self):
        """A column kept on train must be kept on the other splits, or the transform hits a width mismatch."""
        train, _ = _filter_to_numeric(_frame(with_nulls=True))
        val, _ = _filter_to_numeric(_frame(with_nulls=True), keep_cols=list(train.columns))
        assert list(val.columns) == list(train.columns)
        assert pd.api.types.is_numeric_dtype(val["flag"])

    @pytest.mark.parametrize("with_nulls", [True, False])
    def test_both_boolean_shapes_end_up_numeric(self, with_nulls):
        """Nullable or not, a boolean is a usable binary feature and must reach the pipeline."""
        kept, _ = _filter_to_numeric(_frame(with_nulls=with_nulls))
        assert "flag" in kept.columns and pd.api.types.is_numeric_dtype(kept["flag"])
