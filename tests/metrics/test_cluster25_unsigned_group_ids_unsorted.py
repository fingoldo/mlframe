"""Audit 2026-11-20 finding 25-F2: unsorted unsigned group ids are grouped correctly, not read as already sorted."""

import numpy as np

from mlframe.metrics._ranking_extras import hit_at_k


def test_unsorted_unsigned_group_ids_match_signed_ids():
    """uint64 ids in non-monotone order give the same per-query result as the same ids as int64 (np.diff used to wrap on unsigned)."""
    y_true = np.array([1, 0, 0, 1, 0, 1], dtype=np.float64)
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.4], dtype=np.float64)
    gids_signed = np.array([3, 1, 3, 1, 2, 2], dtype=np.int64)
    gids_unsigned = gids_signed.astype(np.uint64)
    assert hit_at_k(y_true, y_score, gids_unsigned, k=1) == hit_at_k(y_true, y_score, gids_signed, k=1)
