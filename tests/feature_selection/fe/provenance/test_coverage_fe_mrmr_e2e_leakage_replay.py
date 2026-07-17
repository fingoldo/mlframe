"""End-to-end MRMR fit -> transform leakage + replay-fidelity contract.

This is the canonical train/serve leakage check at the FULL pipeline level (not the per-
family unit level of the sibling files):

* Fit MRMR with FE on (X_train, y_train); the fit freezes a list of ``EngineeredRecipe`` in
  ``mrmr._engineered_recipes_``.
* Transform a HELD-OUT frame the recipes never saw. Every engineered column the transform
  emits must equal ``apply_recipe(recipe, X_holdout)`` -- i.e. transform replays the frozen
  recipe and nothing else (it does not recompute any fit constant from the held-out data).
* transform()'s optional ``y`` argument must be IGNORED -- passing a wildly wrong y produces
  byte-identical output. There is no path by which a holdout / future target leaks into the
  engineered columns or the selection.
* No frozen recipe captures a y reference (its ``extra`` carries only X-derived constants).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

from tests.feature_selection.conftest import make_fast_mrmr

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def fitted_mrmr_with_fe():
    """Fit a fast MRMR with FE on a synthetic where a multiplicative interaction
    a*b drives a binary target -- FE should fire and freeze recipes."""
    rng = np.random.default_rng(20)
    n = 1200
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=n)
    c = rng.normal(size=n)
    d = rng.normal(size=n)
    signal = a * b + 0.2 * noise
    y = (signal > np.median(signal)).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    # Held-out frame from a fresh seed (different rows, overlapping support).
    rng2 = np.random.default_rng(99)
    Xh = pd.DataFrame(
        {
            "a": rng2.normal(size=300),
            "b": rng2.normal(size=300),
            "c": rng2.normal(size=300),
            "d": rng2.normal(size=300),
        }
    )
    mrmr = make_fast_mrmr(fe=True, interactions_max_order=2)
    mrmr.fit(X, y)
    return mrmr, X, Xh, y


def _engineered_recipes(mrmr):
    return list(getattr(mrmr, "_engineered_recipes_", []) or [])


def test_fit_freezes_engineered_recipes(fitted_mrmr_with_fe):
    mrmr, _X, _Xh, _y = fitted_mrmr_with_fe
    recipes = _engineered_recipes(mrmr)
    # The interaction target should produce at least one engineered recipe.
    assert len(recipes) >= 1, "expected FE to fire on an a*b interaction target"


def test_transform_engineered_columns_equal_recipe_replay(fitted_mrmr_with_fe):
    """Each engineered column in the held-out transform equals apply_recipe on the
    frozen recipe -- transform replays the recipe, computing nothing from holdout y
    and refitting no constant on the holdout frame."""
    mrmr, _X, Xh, _y = fitted_mrmr_with_fe
    recipes = _engineered_recipes(mrmr)
    out = mrmr.transform(Xh)
    assert isinstance(out, pd.DataFrame), "transform should yield a named frame"
    out_cols = set(out.columns)
    checked = 0
    for rec in recipes:
        if rec.name not in out_cols:
            continue  # recipe built but not in final support (dropped by screen)
        replay = np.asarray(apply_recipe(rec, Xh), dtype=np.float64)
        got = out[rec.name].to_numpy(dtype=np.float64)
        # Engineered columns may be stored f32 in the matrix; compare at f32 tol.
        np.testing.assert_allclose(got, replay, rtol=1e-5, atol=1e-5, err_msg=f"recipe {rec.name!r} replay mismatch")
        checked += 1
    assert checked >= 1, "no engineered column reached the transform output to verify"


def test_transform_ignores_y_argument(fitted_mrmr_with_fe):
    """transform(X, y) must ignore y -- a corrupt/holdout y cannot leak into output."""
    mrmr, _X, Xh, _y = fitted_mrmr_with_fe
    out_none = mrmr.transform(Xh)
    bogus_y = np.full(len(Xh), 12345.0)
    out_bogus = mrmr.transform(Xh, bogus_y)
    pd.testing.assert_frame_equal(out_none, out_bogus)


def test_transform_is_deterministic_across_calls(fitted_mrmr_with_fe):
    mrmr, _X, Xh, _y = fitted_mrmr_with_fe
    a = mrmr.transform(Xh)
    b = mrmr.transform(Xh)
    pd.testing.assert_frame_equal(a, b)


def test_no_frozen_recipe_captures_target(fitted_mrmr_with_fe):
    """Defensive: no recipe's extra payload smuggles a per-row y vector (a length-n
    array keyed under a y-ish name would be a leak). Frozen extras hold only
    fit-summary constants (lookups, edges, scalars, small basis params)."""
    mrmr, X, _Xh, _y = fitted_mrmr_with_fe
    n = len(X)
    for rec in _engineered_recipes(mrmr):
        for k, v in dict(rec.extra).items():
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.size == n:
                pytest.fail(f"recipe {rec.name!r} extra[{k!r}] is a length-n ({n}) array -- possible per-row target/feature leak into the frozen recipe")


def test_transform_before_refit_on_new_data_stable(fitted_mrmr_with_fe):
    """Transform on a held-out frame must NOT mutate the fitted recipe state: the
    engineered columns for the original X are unchanged after a holdout transform."""
    mrmr, X, Xh, _y = fitted_mrmr_with_fe
    before = mrmr.transform(X)
    _ = mrmr.transform(Xh)  # serve on new data
    after = mrmr.transform(X)
    pd.testing.assert_frame_equal(before, after)
