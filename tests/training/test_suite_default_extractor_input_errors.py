"""Bad df arguments to the default-extractor path get an actionable error, not an AttributeError (tracker 2026-11-02 wave 16, ERR16-1, ERR16-2)."""

import pathlib

import numpy as np
import polars as pl
import pytest

from mlframe.training.core._main_train_suite import train_mlframe_models_suite


def test_lazyframe_asks_for_collect():
    """A LazyFrame used to fail with 'LazyFrame is not subscriptable ... use select()'."""
    with pytest.raises(TypeError, match="collect"):
        train_mlframe_models_suite(pl.DataFrame({"y": [0, 1]}).lazy(), "y", "m")


def test_list_input_is_a_type_error_naming_the_accepted_types():
    """A list used to fail with \"'list' object has no attribute 'columns'\"."""
    with pytest.raises(TypeError, match="DataFrame"):
        train_mlframe_models_suite([[1, 2], [3, 4]], "y", "m")


def test_csv_path_is_rejected_with_the_parquet_message():
    """A .csv path used to reach pl.read_parquet and fail with a decode or file error."""
    with pytest.raises(ValueError, match="parquet"):
        train_mlframe_models_suite(pathlib.Path("data.csv"), "y", "m")


def test_missing_target_in_a_parquet_file_lists_the_columns(tmp_path):
    """A parquet file without the target column used to raise a raw ColumnNotFoundError."""
    path = tmp_path / "d.parquet"
    pl.DataFrame({"a": np.arange(5), "b": np.arange(5)}).write_parquet(path)
    with pytest.raises(ValueError, match="available"):
        train_mlframe_models_suite(str(path), "y", "m")
