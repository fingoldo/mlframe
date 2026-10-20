"""Consensus aggregation of the tiny-model rerank blocks, pinned on their own return values."""

from __future__ import annotations

import math

from mlframe.training.composite.discovery._tiny_rerank_blocks import (
    _tiny_model_rerank_aggregate_single_per,
    _tiny_model_rerank_consensus_union,
)


def test_union_consensus_keeps_the_best_family_rmse_per_spec():
    """Union consensus keeps, per spec, the lowest RMSE across the families."""
    agg: list[float] = []
    _tiny_model_rerank_aggregate_single_per(["s0", "s1"], ["a", "b"], {"a": [1.0, 4.0], "b": [2.0, 3.0]}, agg, "union")
    assert agg == [1.0, 3.0]


def test_a_spec_with_no_finite_family_rmse_scores_infinity():
    """A spec whose family RMSEs are all non-finite scores infinity, so it never wins a rerank."""
    agg: list[float] = []
    _tiny_model_rerank_aggregate_single_per(["s0", "s1"], ["a", "b"], {"a": [math.nan, 4.0], "b": [math.inf, 3.0]}, agg, "union")
    assert agg == [math.inf, 3.0]


def test_borda_consensus_uses_the_mean_rmse_proxy():
    """Borda consensus uses the mean RMSE as its proxy."""
    agg: list[float] = []
    _tiny_model_rerank_consensus_union("borda", agg, [1.0, 3.0])
    assert agg == [2.0]


def test_unknown_consensus_falls_back_to_the_best_family_rmse():
    """An unrecognised consensus name falls back to the best family RMSE."""
    agg: list[float] = []
    _tiny_model_rerank_consensus_union("other", agg, [2.5, 1.5])
    assert agg == [1.5]
