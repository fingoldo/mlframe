"""Regression: MRMR.transform on a polars frame must raise an actionable error when an
engineered recipe's raw source column is missing, not a raw ``ColumnNotFoundError``.

Pre-fix (fuzz c0126, polars input, custom_prep='pca2'), the polars/ndarray branch of
``_append_engineered`` replayed every recipe via ``apply_recipe(r, X)`` with no check that
``r.src_names`` are actually present in ``X``. When an upstream step (constant-column
removal / outlier detection / custom preprocessing) dropped a raw source column between fit
and transform, ``apply_recipe`` reached into polars internals and raised the native
``polars.exceptions.ColumnNotFoundError: "num_1" not found`` -- an opaque crash pointing at a
polars C-extension frame instead of naming the actual mismatch. The pandas branch already
had this exact guard (see ``test_mrmr_polars_transform_support_remap.py`` for the sibling
``selected_cols`` case); this test pins the same contract for engineered-recipe source
columns on the polars path.
"""

import numpy as np
import polars as pl
import pytest

from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe
from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted_mrmr_with_recipe(names, support, recipe):
    m = MRMR()
    m.feature_names_in_ = np.array(names)
    m.n_features_in_ = len(names)
    m.support_ = np.asarray(support)
    m._engineered_recipes_ = [recipe]
    return m


def test_polars_transform_missing_recipe_source_raises_actionable():
    recipe = EngineeredRecipe(
        name="sqr(num_1)",
        kind="orth_univariate",
        src_names=("num_1",),
        extra={"basis": "raw", "degree": 2},
    )
    # Fit-time saw both "num_0" and "num_1"; support keeps only "num_0" as a base feature, but
    # the engineered recipe still needs "num_1" as its raw operand at replay time.
    m = _fitted_mrmr_with_recipe(["num_0", "num_1"], [0], recipe)
    # Transform-time X is missing "num_1" entirely (dropped upstream between fit and transform).
    X = pl.DataFrame({"num_0": [1.0, 2.0, 3.0]})
    with pytest.raises(RuntimeError, match="missing from input X"):
        m.transform(X)
