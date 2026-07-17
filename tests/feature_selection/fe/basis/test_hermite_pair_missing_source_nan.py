"""Regression: hermite_pair recipe replay NaN-degrades when a source column is missing.

A hermite_pair recipe can reference an engineered intermediate whose producer was pruned at fit time, so it is
absent from the transform frame. `_extract_column` then raised (pandas KeyError / polars ColumnNotFoundError),
crashing the whole transform (fuzz c0006). The mrmr validate-transform contract NaN-degrades chained-capable kinds
for exactly this case; `_apply_hermite_pair` now mirrors it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe, _apply_hermite_pair
from mlframe.feature_selection.filters.hermite_fe import _DEFAULT_BIN_FUNCS, _POLY_BASES


def _recipe_with_missing_source():
    return EngineeredRecipe(
        name="num_present__He1__num_MISSING__He1",
        kind="hermite_pair",
        src_names=("num_present", "num_MISSING"),
        extra={
            "coef_a": np.array([0.0, 1.0], dtype=np.float64),
            "coef_b": np.array([0.0, 1.0], dtype=np.float64),
            "basis": next(iter(_POLY_BASES)),
            "bin_func_name": next(iter(_DEFAULT_BIN_FUNCS)),
            "preprocess_a": {},
            "preprocess_b": {},
            "degree_a": 1,
            "degree_b": 1,
        },
    )


def test_hermite_pair_nan_degrades_on_missing_source_pandas():
    X = pd.DataFrame({"num_present": np.arange(20.0)})  # num_MISSING absent
    out = _apply_hermite_pair(_recipe_with_missing_source(), X)
    assert out.shape == (20,)
    assert np.isnan(out).all()
