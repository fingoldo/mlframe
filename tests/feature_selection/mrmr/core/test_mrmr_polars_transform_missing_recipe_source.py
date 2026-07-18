"""Regression: MRMR.transform on a polars frame handles a missing engineered-recipe source
column the SAME way the pandas branch does, not with an opaque native crash.

Pre-fix (fuzz c0126, polars input, custom_prep='pca2'), the polars/ndarray branch of
``_append_engineered`` replayed every recipe via ``apply_recipe(r, X)`` with no check that
``r.src_names`` are actually present in ``X``. When an upstream step (constant-column
removal / outlier detection / custom preprocessing) dropped a raw source column between fit
and transform, ``apply_recipe`` reached into polars internals and raised the native
``polars.exceptions.ColumnNotFoundError: "num_1" not found`` -- an opaque crash pointing at a
polars C-extension frame instead of naming the actual mismatch.

2026-07 correction: this file originally pinned a hard ``RuntimeError`` for ANY missing recipe
source on polars, added when polars input only had that one (strict, ungraceful) single-pass
replay. Once polars was bridged to the same pandas replay logic (see
``test_mrmr_polars_transform_chained_recipe.py``), that assumption turned out to be an
ACCIDENTAL polars/pandas inconsistency, not a deliberate contract: the pandas branch has ALWAYS
degraded a missing source to a WARNING + neutral zero column for every recipe kind except the
raw-seed-only ones (``_RAW_SEED_ONLY_RECIPE_KINDS`` in ``_mrmr_validate_transform.py``, e.g.
``mi_greedy_transform``) -- verified directly against the unmodified pandas code path. An
``orth_univariate`` recipe (this file's original fixture) is NOT raw-seed-only, so pandas never
raised for it either; polars now matches. The genuinely-actionable-crash contract (raise, name
the column) is pinned below for a RAW-SEED-ONLY kind, where it still fires on both formats.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe
from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted_mrmr_with_recipe(names, support, recipe):
    """Fitted mrmr with recipe."""
    m = MRMR()
    m.feature_names_in_ = np.array(names)
    m.n_features_in_ = len(names)
    m.support_ = np.asarray(support)
    m._engineered_recipes_ = [recipe]
    return m


def test_polars_transform_missing_recipe_source_degrades_like_pandas():
    """Polars transform missing recipe source degrades like pandas."""
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
    X_pl = pl.DataFrame({"num_0": [1.0, 2.0, 3.0]})
    X_pd = pd.DataFrame({"num_0": [1.0, 2.0, 3.0]})

    out_pl = m.transform(X_pl)  # must NOT raise -- matches the pandas degrade contract
    out_pd = m.transform(X_pd)

    np.testing.assert_array_equal(out_pl[recipe.name].to_numpy(), np.zeros(3))
    np.testing.assert_array_equal(out_pd[recipe.name].to_numpy(), np.zeros(3))


def test_polars_transform_missing_raw_seed_only_recipe_source_still_raises():
    """A RAW_SEED_ONLY kind (mi_greedy_transform) never chains -- a missing source there is
    genuine upstream corruption and must still raise, on both formats."""
    from mlframe.feature_selection.filters.engineered_recipes._missingness_ratio_recipes import (
        build_mi_greedy_transform_recipe,
    )

    recipe = build_mi_greedy_transform_recipe(name="log1p(num_1)", transform="log1p", src_names=("num_1",))
    m = _fitted_mrmr_with_recipe(["num_0", "num_1"], [0], recipe)
    X_pl = pl.DataFrame({"num_0": [1.0, 2.0, 3.0]})
    X_pd = pd.DataFrame({"num_0": [1.0, 2.0, 3.0]})

    with pytest.raises(KeyError, match="absent from X"):
        m.transform(X_pl)
    with pytest.raises(KeyError, match="absent from X"):
        m.transform(X_pd)
