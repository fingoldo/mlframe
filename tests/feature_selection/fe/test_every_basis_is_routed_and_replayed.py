"""Every registered basis is chosen by the signal router on data built for it, comes out of the generator as a column, and replays from its frozen recipe to the same values on a row slice."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
    _evaluate_basis_column,
    basis_route_by_signal,
    evaluable_bases,
    generate_univariate_basis_features,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
from mlframe.feature_selection.filters.engineered_recipes._orth_basis_recipes import build_orth_univariate_recipe

from ._basis_cases import DEGREES, N, _case

def test_the_candidates_are_every_registered_basis():
    """No registered basis is left out of the router's candidates."""
    from mlframe.feature_selection.filters.hermite_fe.shared import POLY_BASES

    assert set(evaluable_bases()) == set(POLY_BASES)


@pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev", "laguerre", "fourier", "rbf", "sigmoid", "pade"])
def test_basis_is_routed_generated_and_replayed(basis):
    """The router picks ``basis``; the generator emits its column; the frozen recipe replays the column on a row slice."""
    x, y, degree = _case(basis)
    chosen = basis_route_by_signal(x.astype(np.float32), y, degrees=DEGREES)
    assert chosen == basis, f"routed to {chosen}"

    X = pd.DataFrame({"a": x.astype(np.float32)})
    out = generate_univariate_basis_features(X, degrees=(degree,), basis="auto", y=y, dedup_collinear_sources=False)
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _BASIS_CODE

    name = f"a__{_BASIS_CODE.get(basis, basis)}{degree}"
    assert name in out.columns, f"{name} not in {list(out.columns)}"

    values, params = _evaluate_basis_column(X["a"].to_numpy(), basis, degree, return_params=True)
    recipe = build_orth_univariate_recipe(name="a_cell", src_name="a", basis=basis, degree=degree, preprocess_params=params)
    full = np.asarray(apply_recipe(recipe, X))
    sliced = np.asarray(apply_recipe(recipe, X.iloc[1000:3000]))
    np.testing.assert_array_equal(full[1000:3000], sliced)
    np.testing.assert_allclose(full, values, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out[name].to_numpy(), values, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev", "laguerre", "fourier", "rbf", "sigmoid", "pade"])
def test_family_recipes_are_built_and_replay_for_every_basis(basis):
    """A scoring family pinned to ``basis`` gets its engineered column parsed back into a recipe of that basis, and the recipe replays the column on a row slice."""
    from mlframe.feature_selection.filters._orthogonal_cmim_fe import hybrid_orth_mi_cmim_fe_with_recipes

    x, y, _degree = _case(basis)
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66])) if len(np.unique(y)) > 4 else y.astype(int)
    X = pd.DataFrame({"a": x.astype(np.float32), "b": np.random.default_rng(3).normal(size=N).astype(np.float32)})
    out = hybrid_orth_mi_cmim_fe_with_recipes(X, labels, basis=basis)
    recipes = [r for r in out[2] if r.extra.get("basis") == basis]
    assert recipes, f"no recipe of basis {basis} built; got {[(r.name, r.extra.get('basis')) for r in out[2]]}"
    recipe = recipes[0]
    full = np.asarray(apply_recipe(recipe, X))
    sliced = np.asarray(apply_recipe(recipe, X.iloc[500:2500]))
    assert np.isfinite(full).all()
    np.testing.assert_array_equal(full[500:2500], sliced)
