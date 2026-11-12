"""Every engineered column MRMR selects has a recipe, and the recipe replays it exactly on rows it was not fitted on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


@pytest.fixture(scope="module")
def fitted():
    """One fit with the default FE on a frame whose target has a product, a threshold and a nonlinear term; returns the estimator and the held-out rows."""
    rng = np.random.default_rng(0)
    n, p = 4000, 8
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    y = (np.maximum(X["x2"] - 0.3, 0) ** 2 + 0.5 * X["x0"] * X["x1"] + 0.3 * rng.standard_normal(n) > 0.6).astype(int)
    model = MRMR(verbose=0, random_state=0)
    model.fit(X.iloc[:3000], y.iloc[:3000])
    return model, X.iloc[:3000], X.iloc[3000:]


def test_every_selected_engineered_column_has_a_recipe(fitted):
    """A selected column that is not a raw input must be rebuildable at transform time, so it needs a recipe under its own name."""
    model, X_fit, _ = fitted
    names = list(model.get_feature_names_out())
    engineered = [c for c in names if c not in set(X_fit.columns)]
    assert engineered, "the fit selected no engineered column, so this test would prove nothing"
    recipes = {str(getattr(r, "name", "")) for r in (getattr(model, "_engineered_recipes_", []) or [])}
    assert [c for c in engineered if c not in recipes] == []


def test_transform_returns_the_selected_columns_finite_on_unseen_rows(fitted):
    """The held-out frame comes back with exactly the selected columns, in order, and no non-finite value."""
    model, _, X_new = fitted
    names = list(model.get_feature_names_out())
    out = model.transform(X_new)
    assert list(out.columns) == names
    assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_replay_on_a_row_slice_equals_the_slice_of_the_replay(fitted):
    """The recipes freeze their fit-time statistics, so transforming a slice gives the same values as slicing the transform of the whole frame."""
    model, _, X_new = fitted
    whole = model.transform(X_new)
    part = model.transform(X_new.iloc[:300])
    np.testing.assert_array_equal(part.to_numpy(dtype=float), whole.iloc[:300].to_numpy(dtype=float))
