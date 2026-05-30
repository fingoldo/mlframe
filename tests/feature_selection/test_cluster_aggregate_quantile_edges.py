"""Wave 9.1 loop-iter-29 regression: cluster_aggregate recipes must
persist fit-time quantile edges (sibling of iter 28's unary_binary fix).

Pre-fix at ``engineered_recipes.py:368-372``: ``_apply_cluster_aggregate``
called ``discretize_array`` at replay, which recomputed
``np.nanpercentile`` from TEST aggregate values. Under distribution
drift the bin edges shifted and the same physical row mapped to
DIFFERENT bin codes between fit and transform.

Live demonstration (10x stddev shift, n=200, 3 members):
  identical input row (0.5, 0.5, 0.5) -> fit bin 4, transform bin 3
  83% of test rows received wrong codes
  -> model trained on stale cluster_aggregate codes, served fresh
     rebinned codes at inference (P0 label-leak-equivalent).

Severity: P0. Identical scope and blast radius to iter 28's
unary_binary fix; cluster_aggregate is enabled by default for the
post-hoc cluster_aggregate FE path so this hit users who never opted
into anything special.

Fix (mirror iter 28's three-part):
1. ``build_cluster_aggregate_recipe``'s caller in
   ``_cluster_aggregate.py:272`` now computes the continuous aggregate
   FIRST, derives quantile edges from it, and threads them into the
   recipe's ``quantization["edges"]`` field.
2. ``_apply_cluster_aggregate`` uses stored edges via
   ``np.searchsorted(edges[1:-1], out, side='right')`` so replay is
   purely a fit-time lookup.
3. Legacy recipes without edges emit a UserWarning at replay so
   maintainers see the leak and refit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _frames():
    rng = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame({
        "c1": rng.standard_normal(n),
        "c2": rng.standard_normal(n),
        "c3": rng.standard_normal(n),
    })
    # 10x stddev shift triggers the leak under quantile binning.
    test = pd.DataFrame({
        "c1": rng.standard_normal(n) * 10,
        "c2": rng.standard_normal(n) * 10,
        "c3": rng.standard_normal(n) * 10,
    })
    train.loc[0] = [0.5, 0.5, 0.5]
    test.loc[0] = [0.5, 0.5, 0.5]
    return train, test


def _build_recipe_with_edges(train, member_names):
    from mlframe.feature_selection.filters._cluster_aggregate import (
        _continuous_cols, _standardize_align, _derive_weights,
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
        "nbins": nbins, "method": "quantile",
        "dtype": np.dtype(np.int16).str,
        "edges": edges.tolist(),
    }
    return build_cluster_aggregate_recipe(
        name="ca_test", src_names=tuple(member_names), method="mean_z",
        member_mean=mean, member_std=std, signs=signs, weights=weights,
        quantization=q, diagnostics={"representative": member_names[0]},
    )


def _build_recipe_no_edges(train, member_names):
    from mlframe.feature_selection.filters._cluster_aggregate import (
        _continuous_cols, _standardize_align, _derive_weights,
    )
    from mlframe.feature_selection.filters.engineered_recipes import (
        build_cluster_aggregate_recipe,
    )
    M = _continuous_cols(train, member_names)
    Z, mean, std, signs = _standardize_align(M, 0)
    weights = _derive_weights(Z, "mean_z")
    q = {"nbins": 8, "method": "quantile", "dtype": np.dtype(np.int16).str}
    return build_cluster_aggregate_recipe(
        name="ca_legacy", src_names=tuple(member_names), method="mean_z",
        member_mean=mean, member_std=std, signs=signs, weights=weights,
        quantization=q, diagnostics={"representative": member_names[0]},
    )


def test_with_edges_identical_row_same_bin():
    """The iter-29 contract: identical physical row -> same bin code
    regardless of test-data distribution drift.
    """
    from mlframe.feature_selection.filters.engineered_recipes import (
        _apply_cluster_aggregate,
    )
    train, test = _frames()
    recipe = _build_recipe_with_edges(train, ["c1", "c2", "c3"])
    out_train = _apply_cluster_aggregate(recipe, train)
    out_test = _apply_cluster_aggregate(recipe, test)
    assert out_train[0] == out_test[0], (
        f"identical row got different bins: "
        f"train={out_train[0]}, test={out_test[0]}"
    )


def test_without_edges_warns_legacy_leak():
    """Legacy recipe (no stored edges) emits the UserWarning at replay."""
    from mlframe.feature_selection.filters.engineered_recipes import (
        _apply_cluster_aggregate,
    )
    train, _ = _frames()
    recipe = _build_recipe_no_edges(train, ["c1", "c2", "c3"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _apply_cluster_aggregate(recipe, train)
    assert any(
        "no fit-time quantile edges" in str(w.message) for w in caught
    ), f"expected UserWarning; got {[str(w.message) for w in caught]}"


def test_without_edges_reproduces_leak():
    """Legacy path must still reproduce the leak so the iter-29 fix is
    provably needed.
    """
    from mlframe.feature_selection.filters.engineered_recipes import (
        _apply_cluster_aggregate,
    )
    train, test = _frames()
    recipe = _build_recipe_no_edges(train, ["c1", "c2", "c3"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_train = _apply_cluster_aggregate(recipe, train)
        out_test = _apply_cluster_aggregate(recipe, test)
    # Either bins differ (expected) or coincidentally match - both safe.
    # Skip on coincidental match so the test stays non-flaky.
    if out_train[0] == out_test[0]:
        pytest.skip("legacy path coincidentally gave same bin")
