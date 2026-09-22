"""The uniform calibration binner must cut the prediction range into nbins equal-width bins.

It used a width of span/(nbins-1), so data filled nbins-1 bins and the last one held only the exact maximum: n=200
uniform scores at nbins=10 gave hits [20 25 16 28 17 25 18 21 29 1], a one-row last bin whose observed frequency is 0
or 1 and contributes a near-1.0 gap to CMAEW; at nbins=2 all 200 rows landed in ONE bin.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_plot import calibration_binning


def _bed(n=200, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    return (rng.random(n) < p).astype(int), p


def test_the_last_bin_is_a_real_bin():
    y, p = _bed()
    hits = calibration_binning(y, p, nbins=10, strategy="uniform")[2]
    assert len(hits) == 10
    assert hits[-1] > 5, f"the last bin holds {hits[-1]} row(s); it must hold a tenth of the range, not just the maximum"


def test_two_bins_means_two_populated_bins():
    y, p = _bed()
    hits = calibration_binning(y, p, nbins=2, strategy="uniform")[2]
    assert len(hits) == 2 and hits.sum() == len(p)


@pytest.mark.parametrize("n", [5_000, 300_000])
def test_serial_and_parallel_binners_agree(n):
    """The prange path (large n) carries its own copy of the formula; both must produce the same grid."""
    from mlframe.metrics.calibration._calibration_plot import _fast_calibration_binning_prange, _fast_calibration_binning_serial

    y, p = _bed(n=n, seed=3)
    a = _fast_calibration_binning_serial(y, p, 20)
    b = _fast_calibration_binning_prange(y, p, 20)
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_allclose(a[0], b[0], rtol=1e-12)
