"""Cluster 09 regression: MDL binning can recurse beyond its single top-level split (audit 2026-11-20, finding 09-F25).

``max_depth=1`` (default) keeps the pinned one-split behaviour; a larger depth finds the further edges of a feature whose class changes several times.
"""

import numpy as np

from mlframe.feature_engineering.transformer.mdl_binning_pairwise import _mdl_bin_edges


def _three_blocks():
    """x in three blocks whose class alternates 0 / 1 / 0, so a single split cannot separate them."""
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(0, 1, 200), rng.uniform(2, 3, 200), rng.uniform(4, 5, 200)])
    y = np.concatenate([np.zeros(200), np.ones(200), np.zeros(200)]).astype(np.int32)
    perm = rng.permutation(600)
    return x[perm], y[perm]


def test_default_depth_is_a_single_split():
    """With max_depth=1 at most one edge is returned, however large max_bins is."""
    x, y = _three_blocks()
    assert len(_mdl_bin_edges(x, y, 2, max_bins=20)) <= 1


def test_deeper_search_finds_both_class_boundaries_and_respects_max_bins():
    """max_depth=4 finds an edge in each gap (between 1 and 2, and between 3 and 4) and never exceeds max_bins - 1 edges."""
    x, y = _three_blocks()
    edges = _mdl_bin_edges(x, y, 2, max_bins=20, max_depth=4)
    assert any(1 <= e <= 2 for e in edges) and any(3 <= e <= 4 for e in edges)
    assert len(_mdl_bin_edges(x, y, 2, max_bins=2, max_depth=4)) <= 1
