"""Cluster 40 regression: sanity of the production ranker ``_binned_mi`` that the A/B bench variants are diffed against (audit 2026-11-20, finding 40-F34)."""

import numpy as np

from mlframe.training.ranking._ranker_fs import _binned_mi


def test_dependent_pair_scores_high_independent_pair_low_and_degenerate_inputs_are_zero():
    """y == x gives a large MI (about log(bins) nats), independent noise gives a small one, and constant or too-short input gives exactly 0.0."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    assert _binned_mi(x, x.copy()) > 1.5
    assert _binned_mi(x, rng.normal(size=4000)) < 0.1
    assert _binned_mi(x, np.ones(4000)) == 0.0
    assert _binned_mi(x[:3], x[:3]) == 0.0
