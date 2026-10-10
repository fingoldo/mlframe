"""Make sklearn's HistGradientBoosting see the categorical columns of a polars frame on polars 2.

``categorical_features="from_dtype"`` finds the categorical columns of a non-pandas frame through the dataframe interchange protocol
(``__dataframe__``), which polars 2 removed. Without it the frame is converted as one float array and fails on the first label
(``could not convert string to float``). The columns are named explicitly instead, taken from the polars schema.
"""

from __future__ import annotations

from typing import Any, List

_PENDING_FLAG = "_mlframe_categorical_from_dtype"


def polars_categorical_columns(frame: Any) -> List[str]:
    """Names of the Categorical and Enum columns of a polars frame."""
    import polars as pl

    return [name for name, dtype in frame.schema.items() if dtype == pl.Categorical or dtype == pl.Enum]


def pin_hgb_categorical_features_for_polars(model: Any, frame: Any) -> None:
    """Replace every ``categorical_features="from_dtype"`` of ``model`` (nested estimators included) by the polars frame's categorical names.

    Does nothing for other frames, for polars versions that still have ``__dataframe__`` and for models without that parameter. The
    estimators are remembered, so a later fit on a frame with other columns is pinned again.
    """
    if type(frame).__module__.split(".")[0] != "polars" or hasattr(frame, "__dataframe__") or not hasattr(model, "get_params"):
        return
    names = polars_categorical_columns(frame)
    if not names:
        return
    pinned = {}
    for key, value in model.get_params(deep=True).items():
        if key.split("__")[-1] != "categorical_features":
            continue
        owner = model
        for part in key.split("__")[:-1]:
            owner = getattr(owner, part, None) if owner is not None else None
        pending = isinstance(value, str) and value == "from_dtype"
        if pending or (owner is not None and getattr(owner, _PENDING_FLAG, False)):
            pinned[key] = names
            if owner is not None:
                setattr(owner, _PENDING_FLAG, True)
    if pinned:
        model.set_params(**pinned)
