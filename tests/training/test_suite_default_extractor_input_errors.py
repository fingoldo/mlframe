"""Bad df arguments to the default-extractor path get an actionable error, not an AttributeError (tracker 2026-11-02 wave 16, ERR16-1, ERR16-2)."""

import pathlib

import numpy as np
import polars as pl
import pytest

from mlframe.training.core._main_train_suite import train_mlframe_models_suite


@pytest.mark.parametrize(
    "bad_input,error,match",
    [
        (pl.DataFrame({"y": [0, 1]}).lazy(), TypeError, "collect"),  # used to fail with 'LazyFrame is not subscriptable ... use select()'
        ([[1, 2], [3, 4]], TypeError, "DataFrame"),  # used to fail with "'list' object has no attribute 'columns'"
        (pathlib.Path("data.csv"), ValueError, "parquet"),  # used to reach pl.read_parquet and fail with a decode or file error
    ],
    ids=["lazyframe", "list", "csv_path"],
)
def test_bad_df_argument_gets_an_actionable_error(bad_input, error, match):
    """A LazyFrame, a list and a .csv path are refused with a message that names the accepted input, not an AttributeError or a decode error."""
    with pytest.raises(error, match=match):
        train_mlframe_models_suite(bad_input, "y", "m")


def test_missing_target_in_a_parquet_file_lists_the_columns(tmp_path):
    """A parquet file without the target column used to raise a raw ColumnNotFoundError."""
    path = tmp_path / "d.parquet"
    pl.DataFrame({"a": np.arange(5), "b": np.arange(5)}).write_parquet(path)
    with pytest.raises(ValueError, match="available"):
        train_mlframe_models_suite(str(path), "y", "m")
