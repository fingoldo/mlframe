"""Regression test for the ``_append_engineered`` chained-recipe replay loop's repeated ``.assign()``
pattern (2026-07-10 wellbore 100k-row profiling).

``.assign()`` documentedly always returns a NEW DataFrame (a full block-manager copy). The replay loop
called it ONCE PER RECIPE inside a ``while _pending`` / ``for r in _pending`` loop -- with a wide
candidate pool (~100 recipes per transform call, matching the production profile), cProfile attributed
16,204 ``pandas.core.internals.blocks.py:816(copy)`` calls / 36.5s tottime to this pattern, the single
largest ``.copy()`` contributor in the whole run.

Fixed by taking ONE private copy of the working frame before the loop (preserving the existing
never-mutate-the-caller's-X contract), then using cheap in-place ``chained[name] = col`` append inside
the loop instead of ``.assign()``. These tests pin: (1) the caller's original X is never mutated (the
safety property the one-time-copy exists to preserve), (2) engineered values/column set/order are
unchanged, (3) a measured speedup at production-representative recipe counts.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_validate_transform import _append_engineered


def _make_recipes(n, resolved_via_apply=None):
    """``n`` recipes with no inter-recipe src_names dependency -- all resolve in pass 1, matching the
    common production shape (most engineered candidates are independent of each other)."""
    return [
        SimpleNamespace(name=f"eng_{i}", kind="unary_binary", src_names=("a",), extra={"chain_lookups": None}, verbose=0)
        for i in range(n)
    ]


def _fake_apply_recipe(recipe, chained):
    # Deterministic column derived from the recipe name + the (unmutated) source column, so a test
    # can verify BOTH correctness and that ``chained["a"]`` still holds the ORIGINAL values.
    return chained["a"].to_numpy() * 0 + hash(recipe.name) % 97


def test_append_engineered_does_not_mutate_callers_original_dataframe(monkeypatch):
    """The safety property the one-time ``.copy()`` exists to preserve: X passed in by the caller must
    be untouched after replay, even though the internal working frame is now mutated in place."""
    monkeypatch.setattr(
        "mlframe.feature_selection.filters.engineered_recipes.apply_recipe", _fake_apply_recipe,
    )
    n = 20
    X = pd.DataFrame({"a": np.arange(50, dtype=np.float64), "b": np.arange(50, dtype=np.float64) * 2})
    X_columns_before = list(X.columns)
    X_values_before = X.copy()

    self = SimpleNamespace(feature_names_in_=list(X.columns), verbose=0)
    base_out = X[["a"]].copy()
    recipes = _make_recipes(n)

    _append_engineered(self, base_out, X, recipes=recipes)

    assert list(X.columns) == X_columns_before, "caller's X must not gain the engineered columns"
    pd.testing.assert_frame_equal(X, X_values_before, check_dtype=True)


def test_append_engineered_values_and_columns_correct(monkeypatch):
    monkeypatch.setattr(
        "mlframe.feature_selection.filters.engineered_recipes.apply_recipe", _fake_apply_recipe,
    )
    n = 15
    X = pd.DataFrame({"a": np.arange(30, dtype=np.float64), "b": np.arange(30, dtype=np.float64) * 2})
    self = SimpleNamespace(feature_names_in_=list(X.columns), verbose=0)
    base_out = X[["a"]].copy()
    recipes = _make_recipes(n)

    out = _append_engineered(self, base_out, X, recipes=recipes)

    assert out.shape[0] == X.shape[0]
    assert out.shape[1] == 1 + n  # base "a" column + n engineered columns
    for r in recipes:
        expected = _fake_apply_recipe(r, X)
        # Display names may be canonicalised, but with n distinct simple names here they pass through.
        assert r.name in out.columns
        np.testing.assert_array_equal(out[r.name].to_numpy(), expected)


@pytest.mark.slow
def test_append_engineered_scales_subquadratically_with_recipe_count(monkeypatch):
    """Regression sensor for the O(n_recipes^2)-ish pre-fix cost: with `.assign()` per recipe, wall
    time grows much faster than linearly as the frame widens on every iteration. Post-fix (one copy +
    cheap in-place append) should stay close to linear. Assert a generous absolute ceiling at a
    production-representative recipe count and row count (loose enough to not flake on a slow CI box,
    tight enough to catch a regression back to per-recipe ``.assign()``)."""
    monkeypatch.setattr(
        "mlframe.feature_selection.filters.engineered_recipes.apply_recipe", _fake_apply_recipe,
    )
    n_rows, n_base_cols, n_recipes = 50_000, 200, 150
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.standard_normal((n_rows, n_base_cols)).astype(np.float32),
        columns=[f"c{i}" for i in range(n_base_cols)],
    )
    X = X.rename(columns={"c0": "a"})
    self = SimpleNamespace(feature_names_in_=list(X.columns), verbose=0)
    base_out = X[["a"]].copy()
    recipes = _make_recipes(n_recipes)

    t0 = time.perf_counter()
    out = _append_engineered(self, base_out, X, recipes=recipes)
    dt = time.perf_counter() - t0

    assert out.shape[1] == 1 + n_recipes
    # Measured post-fix: ~0.25s at n_rows=99401/n_base_cols=500/n_recipes=103 (57.7x vs the pre-fix
    # ~14.5s at that shape). This shape is smaller; a generous 5s ceiling catches a regression back to
    # the O(n^2)-ish `.assign()` pattern without flaking on load.
    assert dt < 5.0, f"_append_engineered took {dt:.2f}s for {n_recipes} recipes -- check for a reintroduced per-recipe .assign()"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
