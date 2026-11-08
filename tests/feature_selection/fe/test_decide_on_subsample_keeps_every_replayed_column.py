"""``fe_decide_on_subsample`` rebuilds EVERY engineered column of the subsample decision at full n, not only the last recipe's."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._helpers import fe_decide_on_subsample


def test_all_replayed_columns_reach_the_full_frame(monkeypatch):
    """Three recipes decided on the subsample come back as three full-length columns."""
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = rng.integers(0, 2, n)
    names = ["e1", "e2", "e3"]
    recipes = [SimpleNamespace(name=nm) for nm in names]

    def fit_with_recipes(X_in, y_in, **_kw):
        """The family under test: appends the three engineered columns to the subsample and returns their recipes."""
        out = X_in.copy()
        for nm in names:
            out[nm] = 1.0
        return out, recipes

    values = {nm: np.full(n, float(i + 1)) for i, nm in enumerate(names)}
    monkeypatch.setattr("mlframe.feature_selection.filters.engineered_recipes.apply_recipe", lambda r, _X: values[r.name])
    X_aug, got = fe_decide_on_subsample(fit_with_recipes, X, y, subsample_n=200)
    assert got == recipes
    assert list(X_aug.columns) == ["a", "b", *names]
    for i, nm in enumerate(names):
        assert len(X_aug[nm]) == n
        assert float(X_aug[nm].iloc[0]) == float(i + 1)
