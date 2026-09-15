"""transform() copies only the columns recipes can reach; that set must include columns a recipe reads outside ``src_names``.

The temporal lag / expanding recipes read their time column through ``extra["time_col"]``. The reachable-column walk only inspected ``extra``
when it was a plain ``dict``, and recipe objects hold it in another mapping type, so the time column was left out of the private replay frame
and every temporal recipe raised ``KeyError`` at transform.
"""

from __future__ import annotations

import pandas as pd

from mlframe.feature_selection.filters._mrmr_validate_transform import _recipe_reachable_columns
from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe


def _frame():
    """A frame with more columns than any recipe below needs."""
    return pd.DataFrame({"entity": [1, 1, 2], "x0": [0.1, 0.2, 0.3], "tcol": [0, 1, 2], "unused": [9, 9, 9], "grp2": [1, 2, 3]})


def test_time_column_read_through_extra_is_reachable():
    """A real recipe naming its time column only in ``extra`` keeps that column in the replay frame."""
    recipe = EngineeredRecipe(
        name="tlag1(x0|entity)",
        kind="temporal_lag",
        src_names=("entity", "x0"),
        extra={"entity_cols": ["entity"], "value_col": "x0", "time_col": "tcol", "lag": 1, "global_prior": 0.0},
    )
    cols = _recipe_reachable_columns(_frame(), [recipe])
    assert "tcol" in cols, f"time column dropped from the replay frame: {cols}"
    assert "unused" not in cols, "the narrowing must still leave out columns no recipe names"


def test_column_names_inside_a_list_in_extra_are_reachable():
    """Column names held in a list inside ``extra`` (a group-key list, say) are followed too."""
    recipe = EngineeredRecipe(name="agg(x0|grp2)", kind="grouped_agg", src_names=("x0",), extra={"group_cols": ["grp2"]})
    cols = _recipe_reachable_columns(_frame(), [recipe])
    assert set(cols) == {"x0", "grp2"}
