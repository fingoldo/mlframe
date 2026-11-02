"""A datetime target column is refused with an actionable message instead of being routed to regression (tracker 2026-11-02 wave 16, ERR16-3)."""

import pandas as pd
import polars as pl
import pytest

from mlframe.training.core._main_train_suite_defaults import _build_default_extractor


@pytest.mark.parametrize("frame", [pd.DataFrame({"t": pd.date_range("2024-01-01", periods=5), "x": range(5)}), pl.DataFrame({"t": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 5), eager=True), "x": range(5)})])
def test_datetime_target_raises(frame):
    """The error names the dtype and both ways out."""
    with pytest.raises(ValueError, match="datetime"):
        _build_default_extractor(frame, "t")
