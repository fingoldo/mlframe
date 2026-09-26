"""transform() replays its engineered recipes with one shared column cache and one shared basis cache.

``apply_recipe`` accepts ``col_cache`` / ``basis_cache`` for exactly this loop, but transform passed neither, so a hub source column was
extracted once per recipe and a shared orth operand's basis was evaluated once per recipe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import engineered_recipes
from mlframe.feature_selection.filters.mrmr import MRMR


def _hub_frame(n: int = 3000, seed: int = 0):
    """A target driven by several nonlinear functions of one hub column plus interactions with it."""
    rng = np.random.default_rng(seed)
    hub = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    y = ((hub**2 + np.sin(2 * hub) + hub * b + 0.5 * hub * c + 0.3 * rng.normal(size=n)) > 1.0).astype(np.int64)
    X = pd.DataFrame({"hub": hub, "b": b, "c": c, "noise": rng.normal(size=n)})
    return X, y


def _fit(X, y):
    """A small FE fit that emits engineered recipes."""
    MRMR._FIT_CACHE.clear()
    m = MRMR(random_seed=0, n_jobs=1, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2)
    return m.fit(X, y)


def test_every_recipe_call_receives_the_same_two_caches(monkeypatch):
    """Within one transform call all apply_recipe calls share one col_cache object and one basis_cache object, both non-None."""
    X, y = _hub_frame()
    m = _fit(X, y)
    assert len(getattr(m, "_engineered_recipes_", []) or []) >= 2, "fixture precondition: the fit must emit at least two recipes"
    seen = []
    real = engineered_recipes.apply_recipe

    def _spy(recipe, frame, col_cache=None, basis_cache=None):
        """Record the cache objects each call receives, then replay for real."""
        seen.append((id(col_cache) if col_cache is not None else None, id(basis_cache) if basis_cache is not None else None))
        return real(recipe, frame, col_cache=col_cache, basis_cache=basis_cache)

    monkeypatch.setattr(engineered_recipes, "apply_recipe", _spy)
    m.transform(X)
    assert len(seen) > 0, "transform replayed no recipes"
    assert all(c is not None and b is not None for c, b in seen), f"a recipe was replayed without the caches: {seen}"
    assert len({c for c, _ in seen}) == 1 and len({b for _, b in seen}) == 1, "the caches were not shared across the call"


def test_cached_replay_is_value_identical_and_leaves_input_unchanged(monkeypatch):
    """The cached transform equals a replay with the caches forced off, and the caller's frame is untouched."""
    X, y = _hub_frame(seed=1)
    m = _fit(X, y)
    snapshot = X.copy(deep=True)
    cached = m.transform(X)
    real = engineered_recipes.apply_recipe
    monkeypatch.setattr(engineered_recipes, "apply_recipe", lambda recipe, frame, col_cache=None, basis_cache=None: real(recipe, frame))
    uncached = m.transform(X)
    pd.testing.assert_frame_equal(cached, uncached)
    pd.testing.assert_frame_equal(X, snapshot)
