"""Cluster 11 regression: a numeric RFECV ranking of the wrong length is not read as names (audit 2026-11-20, finding 11-F9).

Numeric ranks over a different feature set used to fall into the name branch, where they were stringified and matched nothing. They now take the
kept-vs-dropped fallback with a WARNING, and a matching-length numeric ranking is still returned as the ranks.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection._benchmarks.fs_hybrid import _arms as arms


def test_mismatched_numeric_ranking_uses_kept_vs_dropped():
    """Three numeric ranks for five names fall back to kept-vs-dropped (11-F9)."""
    names = ["a", "b", "c", "d", "e"]
    model = SimpleNamespace(ranking_=np.array([1, 2, 3]))
    out = arms._rfecv_rank_vector(model, names, ["a", "c"])
    assert out.tolist() == [0.0, 1.0, 0.0, 1.0, 1.0]


def test_matching_numeric_ranking_is_returned_as_ranks():
    """Control: five numeric ranks for five names are returned unchanged (11-F9)."""
    names = ["a", "b", "c", "d", "e"]
    model = SimpleNamespace(ranking_=np.array([3, 1, 2, 5, 4]))
    out = arms._rfecv_rank_vector(model, names, ["a"])
    assert out.tolist() == [3.0, 1.0, 2.0, 5.0, 4.0]
