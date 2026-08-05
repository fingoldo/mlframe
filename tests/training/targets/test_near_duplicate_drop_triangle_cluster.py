"""TRAINING_CORE_A-1 (2026-08-05 audit): the near-duplicate-feature auto-drop's greedy chain-collapse
could drop EVERY column in a 3+ way mutually-correlated cluster (a triangle in the correlation graph, not
just a linear chain), leaving zero survivors, instead of keeping exactly one. Root cause: once either
side of a redundant pair was already in drop_set, the old logic added the OTHER (still-alive) side
instead of treating the edge as already-covered -- so a triangle A-B, B-C, A-C walked to dropping all
three. Fixed by only dropping a column when NEITHER side of the pair is yet in drop_set (a proper
greedy vertex-cover), guaranteeing at least one survivor per connected correlated cluster.
"""

from __future__ import annotations

import pandas as pd

from mlframe.training.core._main_train_suite_target_distribution import (
    _maybe_auto_drop_after_feature_analyzer,
)


class _BehaviorConfig:
    """Minimal stand-in for behavior_config: candidate-list drop off, near-dup drop enabled."""

    auto_drop_distribution_analyzer_candidates = False
    auto_drop_near_duplicate_threshold = 0.9


def _fd_report(pairs):
    """Build a minimal fake feature-distribution report exposing the given redundant_feature_pairs."""

    class _FakeReport:
        """Minimal stand-in for the feature-distribution report exposing only redundant_feature_pairs."""

        drop_candidates: list = []
        diagnostics = {"redundant_feature_pairs": pairs}

    return _FakeReport()


def test_triangle_cluster_keeps_exactly_one_survivor():
    """A, B, C mutually correlated above threshold (a triangle: A-B, B-C, A-C) must NOT all be
    dropped -- exactly one of the three must survive."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3], "keep": [5, 4, 3]})
    pairs = [
        {"a": "a", "b": "b", "corr": 0.99},
        {"a": "b", "b": "c", "corr": 0.99},
        {"a": "a", "b": "c", "corr": 0.99},
    ]
    train_df, _val, _test, dropped = _maybe_auto_drop_after_feature_analyzer(
        fd_report=_fd_report(pairs),
        train_df=df,
        val_df=df,
        test_df=df,
        behavior_config=_BehaviorConfig(),
        metadata={},
        verbose=False,
    )
    survivors = set(train_df.columns)
    cluster_survivors = survivors & {"a", "b", "c"}
    assert len(cluster_survivors) == 1, f"expected exactly one survivor from the {{a,b,c}} triangle, got {cluster_survivors} (dropped={dropped})"
    assert "keep" in survivors, "an unrelated column must never be touched"


def test_larger_fully_connected_cluster_keeps_at_least_one_survivor():
    """A 4-way fully-connected correlated cluster (6 pairwise edges) must also leave at least one
    survivor, regardless of edge processing order."""
    df = pd.DataFrame({"a": [1], "b": [1], "c": [1], "d": [1]})
    pairs = [{"a": x, "b": y, "corr": 0.95} for x, y in [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]]
    train_df, _val, _test, dropped = _maybe_auto_drop_after_feature_analyzer(
        fd_report=_fd_report(pairs),
        train_df=df,
        val_df=df,
        test_df=df,
        behavior_config=_BehaviorConfig(),
        metadata={},
        verbose=False,
    )
    assert len(train_df.columns) >= 1, f"a fully-connected 4-way cluster must not drop every column (dropped={dropped})"


def test_linear_chain_covers_every_correlated_edge():
    """Regression: the original linear-chain case (A-B, B-C) this logic was designed for must still
    end with every correlated EDGE covered (no pair that was flagged redundant has both sides
    survive) -- a vertex cover, not necessarily a single global survivor when the chain's ends
    (A, C here) were never directly flagged as correlated with each other."""
    df = pd.DataFrame({"a": [1], "b": [1], "c": [1]})
    pairs = [{"a": "a", "b": "b", "corr": 0.95}, {"a": "b", "b": "c", "corr": 0.95}]
    train_df, _val, _test, dropped = _maybe_auto_drop_after_feature_analyzer(
        fd_report=_fd_report(pairs),
        train_df=df,
        val_df=df,
        test_df=df,
        behavior_config=_BehaviorConfig(),
        metadata={},
        verbose=False,
    )
    survivors = set(train_df.columns)
    for pair in pairs:
        assert not ({pair["a"], pair["b"]} <= survivors), f"correlated pair {pair['a']!r}/{pair['b']!r} must not both survive, got {survivors}"
    assert len(survivors) >= 1, f"the chain must not drop every column, got dropped={dropped}"
