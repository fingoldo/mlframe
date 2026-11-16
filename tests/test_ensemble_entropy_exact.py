"""Entropy of count rows is exact at the boundaries (single bin, empty row)."""
import numpy as np

from mlframe.feature_engineering.ensemble_features import _entropy_from_counts


def test_single_bin_entropy_is_exactly_zero_and_empty_row_nan():
    """A single-bin row has zero entropy and an empty row is NaN."""
    counts = np.array([[10, 0, 0], [5, 5, 0], [0, 0, 0]], dtype=np.float64)
    h = _entropy_from_counts(counts)
    assert h[0] == 0.0  # old +1e-12 pad gave -1e-12
    assert np.isclose(h[1], np.log(2), atol=1e-15, rtol=0)  # old pad biased this by ~1e-12
    assert np.isnan(h[2])
