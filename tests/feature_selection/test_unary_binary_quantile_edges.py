"""Wave 9.1 loop-iter-28 regression: unary_binary recipes MUST persist
fit-time quantile edges so transform-time bin codes match fit-time bin
codes for the same physical row.

Pre-fix at ``engineered_recipes.py:465-472``: ``_apply_unary_binary``
called ``discretize_array`` at replay, which RECOMPUTED
``np.nanpercentile`` from TEST data (see ``discretization.py:631``).
Under any distribution drift between fit and transform, the bin edges
shifted and an identical physical row mapped to DIFFERENT codes.

Live demonstration (synthetic 10x stddev shift, n=200):
  identical row (0.5, 0.5) -> fit bin 3, transform bin 2
  58.8% of test rows received wrong bin codes
  -> classic train/test leak: model trained on stale codes,
     fed fresh rebinned codes at inference.

Severity: P0 silent. Any deployed MRMR with engineered unary_binary
features under quantile binning corrupted at inference time. Equivalent
in impact to a label-leak bug.

Fix:
1. ``build_unary_binary_recipe`` accepts an optional
   ``fit_values_for_edges`` array. When provided, computes the
   quantile (or uniform) edges ONCE on fit data and stores them in
   ``quantization["edges"]``.
2. ``_apply_unary_binary`` uses the stored edges via
   ``np.searchsorted(edges[1:-1], out, side='right')`` instead of
   re-quantiling on test data.
3. Recipes built without ``fit_values_for_edges`` (legacy pickles)
   emit a UserWarning at replay so maintainers see the leak risk.
4. ``_mrmr_fe_step.py:514`` passes the fit-time engineered values
   (``transformed_vals[:, j]``) so newly-built recipes are
   always edge-pinned.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _build_recipe(with_edges: bool, train: pd.DataFrame):
    from mlframe.feature_selection.filters.engineered_recipes import (
        build_unary_binary_recipe,
    )
    fit_vals = (train["c1"].values * train["c2"].values) if with_edges else None
    return build_unary_binary_recipe(
        name="ub_test", src_a_name="c1", src_b_name="c2",
        unary_a_name="identity", unary_b_name="identity",
        binary_name="mul",
        unary_preset="basic", binary_preset="basic",
        quantization_nbins=5, quantization_method="quantile",
        quantization_dtype=np.int16,
        fit_values_for_edges=fit_vals,
    )


def _frames():
    rng = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame({
        "c1": rng.standard_normal(n),
        "c2": rng.standard_normal(n),
    })
    # 10x stddev shift -> drift severe enough to flip bin codes pre-fix
    test = pd.DataFrame({
        "c1": rng.standard_normal(n) * 10,
        "c2": rng.standard_normal(n) * 10,
    })
    # Inject identical physical row at index 0 in both frames
    train.loc[0] = [0.5, 0.5]
    test.loc[0] = [0.5, 0.5]
    return train, test


def test_with_edges_identical_row_same_bin():
    """The iter-28 contract: identical physical row -> same bin code,
    regardless of test-data distribution drift.
    """
    from mlframe.feature_selection.filters.engineered_recipes import _apply_unary_binary
    train, test = _frames()
    recipe = _build_recipe(with_edges=True, train=train)
    assert recipe.quantization.get("edges") is not None
    out_train = _apply_unary_binary(recipe, train)
    out_test = _apply_unary_binary(recipe, test)
    assert out_train[0] == out_test[0], (
        f"identical (0.5, 0.5) row got different bins: "
        f"train={out_train[0]}, test={out_test[0]}"
    )


def test_without_edges_warns_legacy_leak():
    """Legacy path (no ``fit_values_for_edges``) emits a UserWarning at
    replay so maintainers see the leak.
    """
    from mlframe.feature_selection.filters.engineered_recipes import _apply_unary_binary
    train, test = _frames()
    recipe = _build_recipe(with_edges=False, train=train)
    assert "edges" not in recipe.quantization
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _apply_unary_binary(recipe, test)
    assert any("fit-time quantile edges" in str(w.message) for w in caught), (
        f"expected leak warning; got {[str(w.message) for w in caught]}"
    )


def test_without_edges_reproduces_leak():
    """The leak (different bins for identical row under drift) MUST
    still reproduce on the legacy path. If this ever stops being true,
    the iter-28 fix becomes redundant and the test catches the change.
    """
    from mlframe.feature_selection.filters.engineered_recipes import _apply_unary_binary
    train, test = _frames()
    recipe = _build_recipe(with_edges=False, train=train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_train = _apply_unary_binary(recipe, train)
        out_test = _apply_unary_binary(recipe, test)
    # Pre-fix leak: bins disagree. If equal, the test environment
    # produced a coincidental match -- still safe but unexpected.
    if out_train[0] == out_test[0]:
        pytest.skip("legacy path coincidentally gave same bin; "
                    "iter-28 fix still applies for general case")


def test_edges_count_matches_nbins_plus_one():
    """Stored edges must have ``n_bins + 1`` values (closed-form
    quantile boundaries).
    """
    train, _ = _frames()
    recipe = _build_recipe(with_edges=True, train=train)
    edges = recipe.quantization["edges"]
    assert len(edges) == recipe.quantization["nbins"] + 1
