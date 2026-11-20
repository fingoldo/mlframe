"""A tie in one ranking is not rank agreement (audit 2026-11-20, finding 23-F4)."""
import numpy as np

from mlframe.evaluation.blend_source_selection import check_pairwise_score_correlation


def test_all_tied_second_vector_has_no_agreement():
    """An all-tied b ranks no pair in the same order as a; the agreement must be 0, not the 1.0 the boolean comparison gave."""
    a = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    b = np.ones(5)
    assert check_pairwise_score_correlation(a, b)["rank_agreement"] == 0.0


def test_identical_vectors_agree_fully():
    """Identical orderings still agree on every pair."""
    a = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    assert check_pairwise_score_correlation(a, a)["rank_agreement"] == 1.0
