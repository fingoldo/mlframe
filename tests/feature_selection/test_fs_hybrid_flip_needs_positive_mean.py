"""Audit 2026-11-20 finding 12-F26: a default flips only when the new arm wins most cells AND its mean delta is positive."""

from mlframe.feature_selection._benchmarks.fs_hybrid.round5_innovate_cooccur_clusterrep_bench import decide_flip


def test_majority_of_small_wins_with_negative_mean_does_not_flip():
    """Five tiny wins against two large losses: most cells win, but the mean is negative, so no flip."""
    assert decide_flip([0.001, 0.001, 0.001, 0.001, 0.001, -0.05, -0.05]) is False


def test_majority_wins_with_positive_mean_flips():
    """Clear wins with a positive mean flip the default."""
    assert decide_flip([0.02, 0.03, 0.01, -0.001, 0.02]) is True


def test_no_majority_does_not_flip_even_with_positive_mean():
    """Two wins against two losses is not a majority."""
    assert decide_flip([0.5, 0.5, -0.01, -0.01]) is False
