"""HistGradientBoosting fits on a polars frame with Enum and Categorical columns even when the frame has no __dataframe__ (polars 2)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

from mlframe.training._hgb_polars_categorical import pin_hgb_categorical_features_for_polars, polars_categorical_columns


class _NoInterchangeFrame:
    """Delegates to a polars frame but hides ``__dataframe__``, as polars 2 does."""

    __module__ = "polars.dataframe.frame"

    def __init__(self, frame: pl.DataFrame) -> None:
        """Keep the real frame."""
        self._frame = frame

    def __getattr__(self, name: str):
        """Hide the interchange protocol, delegate the rest."""
        if name == "__dataframe__":
            raise AttributeError(name)
        return getattr(self._frame, name)


def _frame() -> pl.DataFrame:
    """Numeric column plus an Enum and a Categorical column, 60 rows."""
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "x": rng.random(60),
            "e": pl.Series(["a", "b", "c"] * 20).cast(pl.Enum(["a", "b", "c"])),
            "c": pl.Series(["u", "v"] * 30, dtype=pl.Categorical),
        }
    )


def test_categorical_and_enum_columns_are_found_from_the_schema() -> None:
    """Both categorical kinds are named, the numeric column is not."""
    assert polars_categorical_columns(_frame()) == ["e", "c"]


def test_from_dtype_is_replaced_by_the_names_when_the_frame_has_no_interchange_protocol() -> None:
    """The pinned parameter lists the categorical columns, also on an estimator nested in a wrapper."""
    model = MultiOutputClassifier(HistGradientBoostingClassifier(categorical_features="from_dtype"))
    pin_hgb_categorical_features_for_polars(model, _NoInterchangeFrame(_frame()))
    assert model.estimator.categorical_features == ["e", "c"]


def test_a_frame_with_the_interchange_protocol_is_left_alone() -> None:
    """Polars 1 frames keep ``from_dtype``: sklearn handles them itself."""
    if not hasattr(_frame(), "__dataframe__"):
        pytest.skip("this polars has no __dataframe__")
    model = HistGradientBoostingClassifier(categorical_features="from_dtype")
    pin_hgb_categorical_features_for_polars(model, _frame())
    assert model.categorical_features == "from_dtype"


def test_pinned_model_fits_on_a_polars_frame_with_enum_and_categorical_columns() -> None:
    """The fit succeeds and uses the two categorical columns, even through a frame that hides ``__dataframe__``."""
    frame = _frame()
    y = np.tile([0, 1], 30)
    model = HistGradientBoostingClassifier(max_iter=3, categorical_features="from_dtype")
    pin_hgb_categorical_features_for_polars(model, _NoInterchangeFrame(frame))
    model.fit(frame, y)
    assert int(model.is_categorical_.sum()) == 2
