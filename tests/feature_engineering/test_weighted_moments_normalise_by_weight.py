"""Weighted skewness and kurtosis divided by the ROW COUNT instead of the total weight.

The accumulators sum `w_i * d_i**k`, so the weighted moment is that over `sum(w)` -- which is exactly what
`weighted_std` and `weighted_mad` in the same block already divide by. Skew and kurtosis used `size`, scaling
both by `sum_weights / size`. With weights normalised to sum to 1 -- the ordinary case -- that is a factor of n,
and the excess kurtosis then collapses toward the constant -3.0, the same signature a comment in that file
records from an earlier bug in the same lines.

The assertions compare against the textbook weighted standardised moments, so they discriminate: with the old
denominator the values are off by orders of magnitude, not by a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering._numerical_numba import compute_moments_slope_mi
from mlframe.feature_engineering.numerical import get_moments_slope_mi_feature_names


def _moments(arr: np.ndarray, w: np.ndarray) -> dict:
    """Run the kernel and return its outputs keyed by feature name."""
    res, _ = compute_moments_slope_mi(
        arr=arr,
        weights=w,
        mean_value=float(arr.mean()),
        weighted_mean_value=float((arr.astype(np.float64) * w).sum() / w.sum()),
        xvals=None,
        directional_only=False,
        return_lintrend_approx_stats=False,
        compensated=True,
    )
    return dict(zip(get_moments_slope_mi_feature_names(weights=w, directional_only=False, return_lintrend_approx_stats=False), res))


def _reference(arr: np.ndarray, w: np.ndarray):
    """Textbook weighted standardised moments: sum(w*d^k)/sum(w), divided by the weighted std to the k."""
    a = arr.astype(np.float64)
    ww = w.astype(np.float64)
    mu = float((a * ww).sum() / ww.sum())
    d = a - mu
    m2 = float((ww * d**2).sum() / ww.sum())
    skew = float((ww * d**3).sum() / ww.sum()) / m2**1.5
    kurt = float((ww * d**4).sum() / ww.sum()) / m2**2 - 3.0
    return skew, kurt


class TestWeightedMomentsMatchTheDefinition:
    """The denominator is the whole finding, so every assertion here is a value comparison."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_normalised_weights_give_the_reference_values(self, seed):
        """Weights summing to 1 is the worst case: `size` and `sum_weights` differ by n."""
        rng = np.random.default_rng(seed)
        n = 200
        arr = rng.lognormal(0.0, 0.8, n).astype(np.float32)
        w = rng.random(n).astype(np.float32)
        w /= w.sum()
        got = _moments(arr, w)
        ref_skew, ref_kurt = _reference(arr, w)
        assert got["wskew"] == pytest.approx(ref_skew, rel=1e-4), f"weighted skew {got['wskew']} vs reference {ref_skew}"
        assert got["wkurt"] == pytest.approx(ref_kurt, rel=1e-4), f"weighted kurt {got['wkurt']} vs reference {ref_kurt}"

    def test_the_result_is_invariant_to_rescaling_the_weights(self):
        """A standardised weighted moment cannot depend on the weights' scale; with `/size` it did, linearly."""
        rng = np.random.default_rng(3)
        n = 300
        arr = rng.lognormal(0.0, 0.6, n).astype(np.float32)
        w = rng.random(n).astype(np.float32)
        unit = _moments(arr, (w / w.sum()).astype(np.float32))
        scaled = _moments(arr, (w * 1000.0 / w.sum()).astype(np.float32))
        assert unit["wskew"] == pytest.approx(scaled["wskew"], rel=1e-4)
        assert unit["wkurt"] == pytest.approx(scaled["wkurt"], rel=1e-4)

    def test_uniform_weights_reproduce_the_unweighted_moments(self):
        """With every weight equal, the weighted statistic IS the unweighted one -- a fixed point of the fix."""
        rng = np.random.default_rng(4)
        n = 250
        arr = rng.lognormal(0.0, 0.7, n).astype(np.float32)
        got = _moments(arr, np.full(n, 1.0 / n, dtype=np.float32))
        assert got["wskew"] == pytest.approx(got["skew"], rel=1e-3)
        assert got["wkurt"] == pytest.approx(got["kurt"], rel=1e-3)

    def test_the_excess_kurtosis_does_not_collapse_to_minus_three(self):
        """The documented signature of this bug class in this very file."""
        rng = np.random.default_rng(5)
        n = 200
        arr = rng.lognormal(0.0, 1.0, n).astype(np.float32)
        w = rng.random(n).astype(np.float32)
        w /= w.sum()
        assert _moments(arr, w)["wkurt"] > -2.5
