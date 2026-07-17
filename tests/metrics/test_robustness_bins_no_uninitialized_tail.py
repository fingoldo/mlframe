"""Regression: create_robustness_standard_bins must define every returned cell.

Pre-fix, the bin labels came from ``np.empty(npoints)`` and the fill loop only
covered ``step_size * cont_nbins`` cells; when ``npoints % cont_nbins != 0`` (or
``cont_nbins > npoints``) the tail stayed uninitialized garbage. That garbage was
surfaced directly for ``**ORDER**`` and shuffled across all positions for
``**RANDOM**``, making both non-deterministic run-to-run.
"""

import numpy as np
import pytest

from mlframe.metrics._fairness_metrics import create_robustness_standard_bins


@pytest.mark.parametrize("npoints,cont_nbins", [(103, 4), (1000, 7), (10, 3), (5, 8)])
def test_order_bins_deterministic_and_valid(npoints, cont_nbins):
    """Order bins deterministic and valid."""
    b1, uniq = create_robustness_standard_bins("**ORDER**", npoints, cont_nbins)
    b2, _ = create_robustness_standard_bins("**ORDER**", npoints, cont_nbins)

    assert np.array_equal(b1, b2), "ORDER bins must be identical across calls"
    assert b1.shape == (npoints,)
    assert b1.min() >= 0 and b1.max() <= cont_nbins - 1, "no out-of-range (garbage) labels"
    assert set(np.unique(b1).tolist()).issubset(set(uniq))


@pytest.mark.parametrize("npoints,cont_nbins", [(103, 4), (1000, 7), (10, 3)])
def test_random_bins_reproducible_for_same_seed(npoints, cont_nbins):
    """Random bins reproducible for same seed."""
    r1, _ = create_robustness_standard_bins("**RANDOM**", npoints, cont_nbins, seed=0)
    r2, _ = create_robustness_standard_bins("**RANDOM**", npoints, cont_nbins, seed=0)

    assert np.array_equal(r1, r2), "RANDOM bins must be reproducible for a fixed seed"
    assert r1.min() >= 0 and r1.max() <= cont_nbins - 1, "no out-of-range (garbage) labels"


def test_remainder_rows_fall_into_last_bin():
    """Remainder rows fall into last bin."""
    npoints, cont_nbins = 103, 4
    bins, _ = create_robustness_standard_bins("**ORDER**", npoints, cont_nbins)
    step_size = npoints // cont_nbins
    assert np.all(bins[step_size * (cont_nbins - 1) :] == cont_nbins - 1)
