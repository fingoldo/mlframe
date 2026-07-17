"""cluster_aggregate recipe replay: transform emits a CONTINUOUS, leak-free value.

History
-------
Wave 9.1 loop-iter-29 made ``_apply_cluster_aggregate`` persist fit-time quantile
edges and replay them with ``np.searchsorted`` instead of re-quantiling on test
data, fixing a P0 train/test leak (identical physical row -> different bin code
under distribution drift; 83% of rows mis-coded at a 10x stddev shift).

2026-06-12 -- supersession by continuous output
-----------------------------------------------
``_apply_cluster_aggregate`` now returns the aggregate's CONTINUOUS value and no
longer discretises at replay (mirrors the ``unary_binary`` sibling -- see
``_recipe_unary_binary._apply_unary_binary``). Binning the continuous aggregate
to ~10 integer codes keeps only RANK and discards MAGNITUDE, which collapses any
downstream LINEAR model: measured on a target linear in a mean-z cluster
aggregate, a linear model scored test-R2 0.936 on the 10-bin code (Pearson 0.967
with the true aggregate) vs 0.99972 on the continuous value (Pearson 1.000).

Continuous replay SUBSUMES the iter-29 leak fix: the output is a closed-form
function of the operand row given the FROZEN standardization
(``member_mean`` / ``member_std`` / ``signs`` / ``weights``), so an identical
physical row maps to an identical value regardless of the rest of the frame's
distribution -- there is no quantile recomputation left to drift.
``recipe.quantization`` is still BUILT (provenance/audit) but no longer consulted
at replay; the downstream MRMR fit discretises the fit-time column for its OWN MI
matrix via ``_mrmr_fe_step`` (a separate path).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _frames():
    rng = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame(
        {
            "c1": rng.standard_normal(n),
            "c2": rng.standard_normal(n),
            "c3": rng.standard_normal(n),
        }
    )
    # 10x stddev shift -> the drift that flipped bin codes in the pre-iter-29 leak.
    test = pd.DataFrame(
        {
            "c1": rng.standard_normal(n) * 10,
            "c2": rng.standard_normal(n) * 10,
            "c3": rng.standard_normal(n) * 10,
        }
    )
    train.loc[0] = [0.5, 0.5, 0.5]
    test.loc[0] = [0.5, 0.5, 0.5]
    return train, test


def _build_recipe_with_edges(train, member_names):
    from mlframe.feature_selection.filters._cluster_aggregate import (
        _continuous_cols,
        _standardize_align,
        _derive_weights,
    )
    from mlframe.feature_selection.filters.engineered_recipes import (
        build_cluster_aggregate_recipe,
    )

    M = _continuous_cols(train, member_names)
    Z, mean, std, signs = _standardize_align(M, 0)
    weights = _derive_weights(Z, "mean_z")
    agg = Z @ weights
    nbins = 8
    edges = np.nanpercentile(agg, np.linspace(0, 100, nbins + 1))
    q = {
        "nbins": nbins,
        "method": "quantile",
        "dtype": np.dtype(np.int16).str,
        "edges": edges.tolist(),
    }
    return build_cluster_aggregate_recipe(
        name="ca_test",
        src_names=tuple(member_names),
        method="mean_z",
        member_mean=mean,
        member_std=std,
        signs=signs,
        weights=weights,
        quantization=q,
        diagnostics={"representative": member_names[0]},
    ), (mean, std, signs, weights)


def _build_recipe_no_edges(train, member_names):
    from mlframe.feature_selection.filters._cluster_aggregate import (
        _continuous_cols,
        _standardize_align,
        _derive_weights,
    )
    from mlframe.feature_selection.filters.engineered_recipes import (
        build_cluster_aggregate_recipe,
    )

    M = _continuous_cols(train, member_names)
    Z, mean, std, signs = _standardize_align(M, 0)
    weights = _derive_weights(Z, "mean_z")
    q = {"nbins": 8, "method": "quantile", "dtype": np.dtype(np.int16).str}
    return build_cluster_aggregate_recipe(
        name="ca_legacy",
        src_names=tuple(member_names),
        method="mean_z",
        member_mean=mean,
        member_std=std,
        signs=signs,
        weights=weights,
        quantization=q,
        diagnostics={"representative": member_names[0]},
    )


def test_replay_output_is_continuous_not_quantized():
    """Replay emits the continuous mean-z aggregate, not a low-cardinality bin code:
    high cardinality and floating dtype, not integers confined to ``[0, nbins)``."""
    from mlframe.feature_selection.filters.engineered_recipes import _apply_cluster_aggregate

    train, _ = _frames()
    recipe, _ = _build_recipe_with_edges(train, ["c1", "c2", "c3"])
    out = np.asarray(_apply_cluster_aggregate(recipe, train), dtype=np.float64)
    assert np.issubdtype(out.dtype, np.floating)
    assert np.unique(out).size > recipe.quantization["nbins"], (
        f"replay output has only {np.unique(out).size} distinct values -- looks quantized to ~{recipe.quantization['nbins']} bins, not continuous."
    )


def test_identical_row_same_value_under_drift():
    """The leak-safety property, now via CONTINUITY: an identical physical row
    (0.5, 0.5, 0.5) -> identical output whether transformed alongside the
    in-distribution train frame or the 10x-shifted test frame. Continuous replay is a
    closed-form function of the row given the frozen standardization, so there is
    nothing left to drift.
    """
    from mlframe.feature_selection.filters.engineered_recipes import _apply_cluster_aggregate

    train, test = _frames()
    recipe, _ = _build_recipe_with_edges(train, ["c1", "c2", "c3"])
    out_train = _apply_cluster_aggregate(recipe, train)
    out_test = _apply_cluster_aggregate(recipe, test)
    assert out_train[0] == out_test[0], f"identical row got different values: train={out_train[0]}, test={out_test[0]}"


def test_legacy_recipe_without_edges_also_continuous_and_drift_free():
    """A recipe built WITHOUT persisted edges is handled identically now: continuous
    replay, no quantile recomputation, so the old drift-leak cannot occur and NO leak
    warning is emitted (there is no replay-time quantiser left to warn about).
    """
    from mlframe.feature_selection.filters.engineered_recipes import _apply_cluster_aggregate

    train, test = _frames()
    recipe = _build_recipe_no_edges(train, ["c1", "c2", "c3"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_train = _apply_cluster_aggregate(recipe, train)
        out_test = _apply_cluster_aggregate(recipe, test)
    assert not any("fit-time quantile edges" in str(w.message) for w in caught), (
        f"replay no longer quantises, so the legacy-edge leak warning must be gone; got {[str(w.message) for w in caught]}"
    )
    assert out_train[0] == out_test[0], f"continuous replay must give the identical row the same value under drift; train={out_train[0]}, test={out_test[0]}"
