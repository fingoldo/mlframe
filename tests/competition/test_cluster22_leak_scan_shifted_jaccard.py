"""Cluster 22 regression: the shifted Jaccard uses the aligned segments in both terms (audit 2026-11-20, finding 22-F11).

For a shifted comparison the intersection is the overlap of the aligned segments, but the union used the full-row popcounts, so the columns a shift drops
off one end still counted and shifted partners were under-scored. Row b is row a moved one column to the right (one value fell off each end), so at
shift +1 the aligned segments are identical and the overlap is exactly 1.0.
"""

import numpy as np
import pandas as pd

from mlframe.competition.leak_scan import sort_by_density_leak_scan


def test_shifted_partner_scores_one_on_the_aligned_segments():
    """A row and its one-column-shifted copy reach overlap 1.0, not the 0.6 the full-row union gave (22-F11)."""
    df = pd.DataFrame([[0, 1, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0]], columns=list("abcdef"), dtype=float)
    out = sort_by_density_leak_scan(df, overlap_threshold=0.9, min_shared_values=1, max_col_shift=1, neighbor_window=1)
    assert np.isclose(np.nanmax(out["overlap_scores"]), 1.0)
    assert [(a, b) for a, b, _ in out["candidate_pairs"]] == [(0, 1)]
