"""Ranking metrics accept string group ids, sorted or not."""

from __future__ import annotations

import numpy as np

from mlframe.metrics import expected_reciprocal_rank


def test_string_group_ids_match_integer_group_ids():
    """The same grouping written as strings and as integers gives the same ERR, for unsorted rows too."""
    y = np.array([3.0, 0.0, 1.0, 2.0, 0.0, 3.0])
    s = np.array([0.9, 0.1, 0.5, 0.4, 0.3, 0.8])
    as_int = np.array([2, 1, 2, 1, 0, 0])
    as_str = np.array(["g2", "g1", "g2", "g1", "g0", "g0"])
    assert expected_reciprocal_rank(y, s, as_str, k=10) == expected_reciprocal_rank(y, s, as_int, k=10)
