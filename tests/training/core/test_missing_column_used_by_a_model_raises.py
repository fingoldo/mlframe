"""A serving frame missing a column some model reads directly must fail, not warn and predict anyway."""

import pandas as pd
import pytest

from mlframe.training.core._misc_helpers import _validate_input_columns_against_metadata


def _metadata(schema_names):
    return {
        "raw_input_columns": ["a", "b", "c"],
        "model_schemas": {"m.dump": {"schema_hash": "h", "input_schema": [{"name": n, "role": "num", "dtype": "float64"} for n in schema_names]}},
    }


def test_a_column_a_model_reads_directly_is_a_hard_failure():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})  # "c" is gone
    with pytest.raises(ValueError, match=r"c \(used by m.dump\)"):
        _validate_input_columns_against_metadata(df, _metadata(["a", "b", "c"]))


def test_a_column_no_model_reads_still_only_warns(caplog):
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with caplog.at_level("WARNING", logger="mlframe.training.core._misc_helpers"):
        out = _validate_input_columns_against_metadata(df, _metadata(["a", "b"]))
    assert list(out.columns) == ["a", "b"]
    assert any("Missing columns in input" in r.getMessage() for r in caplog.records)


def test_post_pipeline_columns_never_trigger_it():
    """The model schema is a POST-pipeline snapshot; names the pipeline creates are absent from the serving frame by design."""
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    out = _validate_input_columns_against_metadata(df, _metadata(["a", "b", "c", "pca0", "pca1"]))
    assert list(out.columns) == ["a", "b", "c"]
