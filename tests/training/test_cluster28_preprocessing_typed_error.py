"""Cluster 28 regression: load_and_prepare_dataframe raises a typed error with a message, not a bare AssertionError (audit 2026-11-20, finding 28-F52).

A bare assertion is stripped under ``python -O`` and carries no message. The loader must raise a TypeError that names the function and the type
it received.
"""

import polars as pl
import pytest

from mlframe.training import preprocessing as pp


def test_non_dataframe_after_collection_raises_typed_error_naming_function_and_type(monkeypatch):
    """A collection step that returns something other than a polars DataFrame raises TypeError with the function name and the received type (28-F52)."""
    monkeypatch.setattr(pp, "_collect_streaming", lambda _df: "not a frame")
    frame = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(TypeError, match=r"load_and_prepare_dataframe: expected a polars DataFrame after collection, got str"):
        pp.load_and_prepare_dataframe(frame, pp.PreprocessingConfig(), verbose=0)
