"""The compiled ``pointwise_mi`` equals its numpy form and leaves out-of-range codes to it."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_gain_stats as gain


def test_compiled_values_equal_the_numpy_form():
    """Random code vectors of both integer widths, some with an empty bin: the two forms agree to 1e-12 and the mean is the plug-in MI."""
    rng = np.random.default_rng(0)
    for trial in range(120):
        n = int(rng.integers(1, 8000))
        n_codes, ky = int(rng.integers(2, 13)), int(rng.integers(2, 6))
        codes = rng.integers(0, n_codes if trial % 3 else max(1, n_codes - 1), n).astype(np.int64 if trial % 2 else np.int32)
        y = rng.integers(0, ky, n)
        got = gain.pointwise_mi(codes, n_codes, y, ky)
        want = gain._pointwise_mi_numpy(codes, n_codes, y, ky)
        assert got.dtype == want.dtype and got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
        assert gain.plugin_mi_of_codes(codes, n_codes, y, ky) == pytest.approx(float(want.mean()), abs=1e-12)


def test_out_of_range_code_raises_as_before():
    """A code outside ``[0, n_codes)`` is refused by the compiled pass and reaches the numpy form, which raises."""
    with pytest.raises(ValueError):
        gain.pointwise_mi(np.array([0, 5]), 3, np.array([0, 1]), 2)


def test_dependent_codes_score_above_independent_ones():
    """Codes that determine the target have the target's entropy as MI; independent codes score near zero."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 6000)
    assert gain.plugin_mi_of_codes(y.copy(), 2, y, 2) == pytest.approx(np.log(2), abs=5e-3)
    assert gain.plugin_mi_of_codes(rng.integers(0, 10, 6000), 10, y, 2) < 5e-3
