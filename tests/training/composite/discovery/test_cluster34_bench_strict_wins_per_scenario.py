"""Audit 2026-11-20 finding 34-F33: the strict-win counts are per scenario, so the default's count is not inflated.

Two challengers both lose to the default in one scenario: the default counts once for that scenario, not twice.
"""

from mlframe.training.composite.discovery._benchmarks import bench_mi_bin_nbins_ranking as bench


def _row(conc):
    """A results entry with the given concordance."""
    return {"concordance": conc, "spearman": 0.0}


def test_default_counts_once_per_scenario_not_once_per_challenger():
    """Default 0.9 beats 4 and 8 (both 0.5) in one scenario: the default's count is 1, each challenger's is 0."""
    grid = (4, 8, 16)
    res = {"s|nbins=16": _row(0.9), "s|nbins=4": _row(0.5), "s|nbins=8": _row(0.5)}
    counts = bench._strict_win_counts(res, ["s"], grid, 0.005)
    assert counts[16] == 1
    assert counts[4] == 0 and counts[8] == 0


def test_challenger_win_counts_once_per_scenario():
    """A challenger that beats the default by more than the margin counts 1 in that scenario."""
    grid = (4, 16)
    res = {"s|nbins=16": _row(0.5), "s|nbins=4": _row(0.9)}
    counts = bench._strict_win_counts(res, ["s"], grid, 0.005)
    assert counts[4] == 1 and counts[16] == 0
