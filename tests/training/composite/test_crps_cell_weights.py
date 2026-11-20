"""CRPS quantile decomposition weights follow the cell width of the level grid (audit 2026-11-20, finding 35-F2)."""
import numpy as np

from mlframe.training.composite.distributional import _crps_cell_weights


def test_midpoint_grid_keeps_the_two_over_k_weight():
    """A midpoint grid (k - 0.5) / K has cells of width 1 / K, so the weights stay 2 / K as before."""
    K = 19
    levels = (np.arange(1, K + 1) - 0.5) / K
    np.testing.assert_allclose(_crps_cell_weights(levels), np.full(K, 2.0 / K))


def test_uneven_grid_weights_are_its_cell_widths_and_integrate_to_two():
    """An uneven grid gets its own cell widths, which sum to 2 and are not uniform."""
    levels = np.linspace(0.1, 0.9, 9)  # the grid used by quantiles=[0.1..0.9]
    w = _crps_cell_weights(levels)
    np.testing.assert_allclose(w.sum(), 2.0)
    # Interior levels own a cell of width 0.1; the two outer levels own cells of width 0.15.
    np.testing.assert_allclose(w[1:-1], np.full(7, 0.2))
    np.testing.assert_allclose(w[[0, -1]], np.full(2, 0.3))
    # The old uniform 2 / K weight (0.222...) gives every level the same weight, which an uneven grid does not have.
    assert not np.allclose(w, np.full(9, 2.0 / 9))
