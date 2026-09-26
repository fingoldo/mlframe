"""A pure form whose recipe cannot be replayed at fit is not retained.

The engineered-subsumption guard replays each retention candidate with ``apply_recipe``, the same call ``transform()``
makes. When that replay raised, the candidate was retained "conservatively", which shipped a column the fitted selector
could not produce at predict time.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class _UnreplayableRecipe:
    """A recipe no dispatcher knows: ``apply_recipe`` raises on it, at fit and at transform alike."""

    name = "unreplayable_pure_form"
    src_names = ("a", "b")


def _case(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = rng.random(n) + 0.5, rng.random(n) + 0.5, rng.random(n) + 0.5
    d, e = rng.random(n), rng.random(n)
    y = 0.2 * a**2 / b + np.log(c * 2.0) * np.sin(d / 3.0)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), y


def test_a_retention_candidate_that_cannot_replay_is_not_kept(monkeypatch):
    from mlframe.feature_selection.filters import MRMR, _fe_pure_form_retention
    from mlframe.feature_selection.filters.engineered_recipes import _recipe_dispatch

    replay_attempts = []
    real_apply = _recipe_dispatch.apply_recipe

    def spy(recipe, *a, **k):
        if isinstance(recipe, _UnreplayableRecipe):
            replay_attempts.append(recipe.name)
        return real_apply(recipe, *a, **k)

    monkeypatch.setattr(_recipe_dispatch, "apply_recipe", spy)
    monkeypatch.setattr(_fe_pure_form_retention, "retain_usable_pure_forms", lambda *a, **k: [(_UnreplayableRecipe(), _UnreplayableRecipe.name)])
    df, y = _case()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(random_seed=0).fit(X=df, y=y)
    assert replay_attempts, "setup: the subsumption guard never replayed the candidate (no engineered incumbent to condition on)"
    names = [getattr(r, "name", None) for r in (getattr(fs, "_engineered_recipes_", None) or [])]
    assert _UnreplayableRecipe.name not in names
