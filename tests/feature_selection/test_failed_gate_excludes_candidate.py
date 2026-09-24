"""A candidate an enabled gate could not evaluate is excluded, not ranked un-gated beside gated candidates."""

import numpy as np

from mlframe.feature_selection.filters.evaluation import _exclude_ungated_candidate


def test_the_candidate_is_zeroed_everywhere_the_ranking_reads():
    partial = {7: (0.4, 3)}
    expected = np.array([0.1] * 10)
    gain = _exclude_ungated_candidate(7, partial, expected)
    assert gain == 0.0
    assert partial[7] == (0.0, 3), "the partial gain keeps its count but loses the un-gated value"
    assert expected[7] == 0.0 and expected[6] == 0.1
