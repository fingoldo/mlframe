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

N = 6000
DEGREES = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _case(basis: str):
    """``(x, y, expected degree)``: a column and a target that the ``basis`` expansion of that degree reproduces better than any other basis."""
    rng = np.random.default_rng(11)
    if basis == "hermite":
        x = rng.normal(size=N)
        return x, x**2 - 1.0, 2
    if basis == "legendre":
        x = rng.uniform(-1, 1, N)
        return x, (3 * x**2 - 1) / 2, 2
    if basis == "chebyshev":
        x = rng.uniform(-1, 1, N)
        return x, 4 * x**3 - 3 * x, 3
    if basis == "laguerre":
        x = rng.exponential(size=N)
        return x, (x**2 - 4 * x + 2) / 2, 2
    if basis == "fourier":
        x = rng.uniform(0, 1, N)
        return x, np.sin(2 * np.pi * x), 1
    if basis == "rbf":
        x = rng.normal(size=N)
        centre = np.quantile(x, 0.8)
        return x, (np.abs(x - centre) < 0.12).astype(float), 8
    if basis == "sigmoid":
        x = rng.uniform(0, 1, N)
        return x, (x > np.quantile(x, 0.9)).astype(float), 9
    if basis == "pade":
        x = rng.normal(size=N)
        z = (x - x.mean()) / x.std()
        return x, 1.0 / (1.0 + z**2), 2
    raise KeyError(basis)


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
