"""Tied quantile edges are raised above their predecessor even when 1e-9 is below one float64 step."""
import numpy as np

from mlframe.feature_engineering.transformer._utils import strictly_increasing_edges


def test_edges_at_offset_1e12_are_strictly_increasing():
    """At 1e12 the float64 spacing is about 1e-4, so adding 1e-9 left the edges equal and the buckets empty."""
    edges = strictly_increasing_edges(np.array([1e12, 1e12, 1e12, 1e12 + 5.0]))
    assert np.all(np.diff(edges) > 0.0)


def test_small_scale_edges_still_move_by_one_nano():
    """At ordinary scale the historical 1e-9 nudge is kept."""
    edges = strictly_increasing_edges(np.array([0.0, 0.0, 1.0]))
    assert abs(edges[1] - 1e-9) < 1e-15
