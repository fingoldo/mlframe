"""The compiled held-out cubic R^2 equals the Vandermonde least-squares form it replaces."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe as ebf


def _reference(x: np.ndarray, prep: tuple) -> float:
    """The numpy form: rank-normalise, fit the cubic on the training rows by least squares, score the validation rows."""
    from scipy.stats import rankdata

    va, tr, y_tr, yv, sst = prep
    zx = (rankdata(x, method="average") - 1.0) / max(x.size - 1, 1) * 2.0 - 1.0
    coef, *_ = np.linalg.lstsq(np.vander(zx[tr], 4), y_tr, rcond=None)
    return float(1.0 - np.sum((yv - np.vander(zx[va], 4) @ coef) ** 2) / sst)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["smooth", "noise", "ties"])
def test_fast_r2_equals_the_least_squares_reference(dtype, kind):
    """A cubic signal, pure noise and a column with many tied values: the same R^2 to 1e-6."""
    rng = np.random.default_rng(3)
    n = 6000
    x = rng.standard_normal(n)
    y = {"smooth": x**3 - 2 * x + 0.3 * rng.standard_normal(n), "noise": rng.standard_normal(n), "ties": np.round(x, 1) ** 2 + 0.1 * rng.standard_normal(n)}[kind]
    if kind == "ties":
        x = np.round(x, 1)
    x = x.astype(dtype)
    prep = ebf._heldout_smooth_r2_prep(y)
    got = ebf._heldout_smooth_r2_fast(x, prep)
    assert abs(got - _reference(x, prep)) < 1e-6
    assert got < 0.05 if kind == "noise" else got > 0.3  # the cubic recovers a smooth signal and finds nothing in noise


def test_singular_systems_and_degenerate_columns_answer_as_before():
    """A column with two distinct values makes the normal system singular and takes the least-squares route; a constant or non-finite column scores 0."""
    rng = np.random.default_rng(4)
    y = rng.standard_normal(500)
    prep = ebf._heldout_smooth_r2_prep(y)
    two = (rng.random(500) < 0.5).astype(np.float64)
    assert abs(ebf._heldout_smooth_r2_fast(two, prep) - _reference(two, prep)) < 1e-6
    assert ebf._heldout_smooth_r2_fast(np.ones(500), prep) == 0.0
    assert ebf._heldout_smooth_r2_fast(np.where(np.arange(500) == 3, np.nan, 1.0), prep) == 0.0
