"""Regression: MRMR.transform on a polars frame must resolve a CHAINED engineered recipe
(one whose ``src_names`` reference another recipe's OUTPUT name, not a raw input column).

Pre-fix, ``_append_engineered``'s polars/ndarray branch replayed every recipe with a single
best-effort pass that validated ``r.src_names`` against the raw input columns only -- it had no
multi-pass topological resolution, so ANY recipe chained on a sibling recipe's output raised
``RuntimeError: ... references source column(s) [...] missing from input X`` even though the
source was perfectly reconstructable (the pandas branch already does this multi-pass replay).

Confirmed live (2026-07): ``CompositeCrossTargetEnsemble``'s k-fold OOF refit passes the suite's
polars ``train_df`` through ``MRMR.transform``, so on a real 4.1M-row wellbore fit where MRMR
selected a chained ``add(add(neg(A),abs(B)),mul(reciproc(C),abs(D)))`` recipe (fe_max_steps=2
default), EVERY k-fold refit attempt AND the ``CompositeMoE`` expert evaluation failed for the
whole target with this exact error text -- silently disabling the honest-OOF ensemble weighting
and MoE gate. Fixed by bridging polars input to the project's sanctioned zero-copy
``get_pandas_view_of_polars_df`` view before recipe replay, reusing the SAME multi-pass
resolution the pandas branch already has (see ``_mrmr_validate_transform.py``).
"""

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe
from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted_mrmr(names, support, recipes):
    m = MRMR()
    m.feature_names_in_ = np.array(names)
    m.n_features_in_ = len(names)
    m.support_ = np.asarray(support)
    m._engineered_recipes_ = list(recipes)
    return m


def _build_chained_recipes():
    # Two leaf recipes over raw columns, mirroring the production shape
    # add(neg(A), abs(B)) and mul(reciproc(C), abs(D)).
    recipe_a = EngineeredRecipe(
        name="add(neg(A),abs(B))",
        kind="unary_binary",
        src_names=("A", "B"),
        unary_names=("neg", "abs"),
        binary_name="add",
    )
    recipe_b = EngineeredRecipe(
        name="mul(reciproc(C),abs(D))",
        kind="unary_binary",
        src_names=("C", "D"),
        unary_names=("reciproc", "abs"),
        binary_name="mul",
    )
    # The CHAINED recipe: its two operands are the two recipes above, by NAME -- not raw columns.
    recipe_c = EngineeredRecipe(
        name="add(add(neg(A),abs(B)),mul(reciproc(C),abs(D)))",
        kind="unary_binary",
        src_names=(recipe_a.name, recipe_b.name),
        unary_names=("identity", "identity"),
        binary_name="add",
    )
    return recipe_a, recipe_b, recipe_c


def _sample_frame():
    rng = np.random.default_rng(0)
    n = 50
    return {
        "A": rng.normal(size=n),
        "B": rng.normal(size=n),
        "C": rng.uniform(1.0, 5.0, size=n),  # avoid reciproc(0)
        "D": rng.normal(size=n),
    }


def test_polars_transform_resolves_chained_recipe():
    """Pre-fix this raised RuntimeError; post-fix it must succeed and match the pandas path."""
    data = _sample_frame()
    recipe_a, recipe_b, recipe_c = _build_chained_recipes()
    m = _fitted_mrmr(["A", "B", "C", "D"], [0, 1, 2, 3], [recipe_a, recipe_b, recipe_c])

    X_pd = pd.DataFrame(data)
    X_pl = pl.DataFrame(data)

    out_pd = m.transform(X_pd)
    out_pl = m.transform(X_pl)  # must NOT raise

    assert isinstance(out_pl, pl.DataFrame)
    np.testing.assert_array_equal(out_pd["A"].to_numpy(), out_pl["A"].to_numpy())

    expected_a = -data["A"] + np.abs(data["B"])
    expected_b = (1.0 / data["C"]) * np.abs(data["D"])
    expected_c = expected_a + expected_b

    np.testing.assert_allclose(out_pd[recipe_a.name].to_numpy(), expected_a)
    np.testing.assert_allclose(out_pd[recipe_c.name].to_numpy(), expected_c)
    np.testing.assert_allclose(out_pl[recipe_a.name].to_numpy(), expected_a)
    np.testing.assert_allclose(out_pl[recipe_c.name].to_numpy(), expected_c)

    # Cross-format parity: polars and pandas replay must agree exactly on every engineered column.
    for r in (recipe_a, recipe_b, recipe_c):
        np.testing.assert_array_equal(out_pd[r.name].to_numpy(), out_pl[r.name].to_numpy())


def test_polars_transform_chained_recipe_column_names_match_get_feature_names_out():
    """The polars output must use the SAME simplified display names as get_feature_names_out
    and the pandas branch -- pre-fix the polars branch named columns via the raw ``r.name``."""
    data = _sample_frame()
    recipe_a, recipe_b, recipe_c = _build_chained_recipes()
    m = _fitted_mrmr(["A", "B", "C", "D"], [0, 1, 2, 3], [recipe_a, recipe_b, recipe_c])

    out_pl = m.transform(pl.DataFrame(data))
    expected_names = list(m.get_feature_names_out())
    assert list(out_pl.columns) == expected_names
