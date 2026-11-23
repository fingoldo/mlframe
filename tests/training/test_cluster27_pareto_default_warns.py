"""Cluster 27 regression: when no frontier point is scored, the default first iteration is returned with a WARNING (audit 2026-11-20, finding 27-F18)."""

import logging

from mlframe.training import _cv_aggregation as agg


def test_unscored_frontier_warns_and_returns_first(caplog):
    """Frontier points with no shard scores and non-finite mean/std give the first index, and say so (27-F18)."""
    nan = float("nan")
    with caplog.at_level(logging.WARNING, logger=agg.logger.name):
        out = agg.select_from_pareto([3, 5], iter_means=[nan] * 6, iter_stds=[nan] * 6, iter_shard_scores=[[] for _ in range(6)], direction="min")
    assert out == 3
    assert any("no frontier point had shard scores" in r.getMessage() for r in caplog.records)
