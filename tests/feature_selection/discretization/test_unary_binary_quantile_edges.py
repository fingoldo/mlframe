"""unary_binary recipe replay: transform emits a CONTINUOUS, leak-free value.

History
-------
Wave 9.1 loop-iter-28 made ``_apply_unary_binary`` persist fit-time quantile
edges and replay them with ``np.searchsorted`` instead of re-quantiling on test
data. That fixed a P0 train/test leak: pre-iter-28 the replay recomputed
``np.nanpercentile`` from TEST data, so under distribution drift an identical
physical row mapped to DIFFERENT bin codes between fit and transform (58.8% of
rows mis-coded at a 10x stddev shift).

2026-06-12 -- supersession by continuous output
-----------------------------------------------
``_apply_unary_binary`` now returns the engineered column's CONTINUOUS value and
no longer discretises at replay at all (see the rationale in
``_recipe_unary_binary._apply_unary_binary``: a heavy-tailed product binned to
~10 integer codes keeps only RANK and discards MAGNITUDE, which collapses any
downstream LINEAR model -- measured test-R2 ~0.002 on ``y=0.2*a**2/b`` via the
10-bin code vs >=0.99 on the continuous feature). This is the ``prewarp`` /
``hermite_pair`` siblings' behaviour generalised to every unary_binary recipe.

Continuous replay SUBSUMES the iter-28 leak fix: the output is now a pure
element-wise (closed-form) function of the operand row, so an identical physical
row maps to an identical value REGARDLESS of the rest of the frame's distribution
-- there is no quantile recomputation left to drift. These tests therefore assert
the stronger, post-supersession contract: continuous, magnitude-preserving, and
drift-invariant. ``quantization['edges']`` is still BUILT (provenance/audit) but
is no longer consulted at replay; the downstream MRMR fit discretises the
fit-time column for its OWN MI matrix via ``_mrmr_fe_step`` (a separate path).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


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
    # 10x stddev shift -> the drift that flipped bin codes in the pre-iter-28 leak.
    test = pd.DataFrame({
        "c1": rng.standard_normal(n) * 10,
        "c2": rng.standard_normal(n) * 10,
    })
    # Inject identical physical row at index 0 in both frames.
    train.loc[0] = [0.5, 0.5]
    test.loc[0] = [0.5, 0.5]
    return train, test


def test_replay_output_is_continuous_not_quantized():
    """The replay emits the continuous product value, not a low-cardinality bin
    code. For ``mul(identity(c1), identity(c2))`` the column equals ``c1*c2``
    exactly (float, high cardinality), never an integer code in ``[0, nbins)``.
    """
    from mlframe.feature_selection.filters.engineered_recipes import _apply_unary_binary
    train, _ = _frames()
    recipe = _build_recipe(with_edges=True, train=train)
    out = np.asarray(_apply_unary_binary(recipe, train), dtype=np.float64)
    expected = train["c1"].to_numpy() * train["c2"].to_numpy()
    # Continuous: matches the closed-form product, not a 5-bin code.
    assert np.allclose(out, expected, atol=1e-9), (
        f"replay should equal the continuous product c1*c2; max abs diff "
        f"{np.max(np.abs(out - expected)):.3e}, sample out[:5]={out[:5]}"
    )
    assert np.unique(out).size > recipe.quantization["nbins"], (
        f"replay output has only {np.unique(out).size} distinct values -- looks "
        f"quantized to ~{recipe.quantization['nbins']} bins, not continuous."
    )


def test_identical_row_same_value_under_drift():
    """The leak-safety property, now via CONTINUITY: an identical physical row
    (0.5, 0.5) -> identical output whether transformed alongside the in-distribution
    train frame or the 10x-shifted test frame. Continuous replay is a closed-form
    function of the row alone, so there is nothing left to drift.
    """
    from mlframe.feature_selection.filters.engineered_recipes import _apply_unary_binary
    train, test = _frames()
    recipe = _build_recipe(with_edges=True, train=train)
    out_train = _apply_unary_binary(recipe, train)
    out_test = _apply_unary_binary(recipe, test)
    assert out_train[0] == out_test[0], (
        f"identical (0.5, 0.5) row produced different values: "
        f"train={out_train[0]}, test={out_test[0]}"
    )
    assert np.isclose(float(out_train[0]), 0.25, atol=1e-9), (
        f"mul(0.5, 0.5) should be 0.25 continuous; got {out_train[0]}"
    )


def test_legacy_recipe_without_edges_also_continuous_and_drift_free():
    """A recipe built WITHOUT ``fit_values_for_edges`` (legacy pickle shape) is now
    handled identically: continuous replay, no quantile recomputation, so the old
    drift-leak cannot occur and NO leak warning is emitted (there is no replay-time
    quantiser left to warn about).
    """
    from mlframe.feature_selection.filters.engineered_recipes import _apply_unary_binary
    train, test = _frames()
    recipe = _build_recipe(with_edges=False, train=train)
    assert "edges" not in recipe.quantization
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_train = _apply_unary_binary(recipe, train)
        out_test = _apply_unary_binary(recipe, test)
    assert not any("fit-time quantile edges" in str(w.message) for w in caught), (
        f"replay no longer quantises, so the legacy-edge leak warning must be gone; "
        f"got {[str(w.message) for w in caught]}"
    )
    # The pre-iter-28 leak (identical row -> different code under drift) is structurally
    # eliminated: identical row -> identical continuous value.
    assert out_train[0] == out_test[0], (
        f"continuous replay must give the identical row the same value under drift; "
        f"train={out_train[0]}, test={out_test[0]}"
    )


def test_edges_still_built_as_provenance():
    """``build_unary_binary_recipe`` still derives ``n_bins + 1`` fit-time quantile
    edges and stores them on the recipe. They are provenance/audit only now (replay
    no longer consults them), but the build contract is unchanged.
    """
    train, _ = _frames()
    recipe = _build_recipe(with_edges=True, train=train)
    edges = recipe.quantization["edges"]
    assert len(edges) == recipe.quantization["nbins"] + 1
