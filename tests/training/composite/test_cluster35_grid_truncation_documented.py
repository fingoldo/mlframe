"""Cluster 35 regression: the default quantile grid's tail truncation is documented at the grid (audit 2026-11-20, finding 35-F28).

The default grid spans 0.05 .. 0.95, so sampling and CRPS ignore the outer 5% per side. The choice and the way to remove it (an explicit tail
grid) are stated where the grid is defined.
"""

from mlframe.training.composite import distributional


def test_default_grid_is_still_the_dense_0_05_to_0_95():
    """Control: the default grid is unchanged, 19 levels from 0.05 to 0.95 (35-F28)."""
    grid = distributional._DEFAULT_DENSE_QUANTILES
    assert len(grid) == 19
    assert grid[0] == 0.05 and grid[-1] == 0.95
