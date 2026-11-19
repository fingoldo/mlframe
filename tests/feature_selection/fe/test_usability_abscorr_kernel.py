"""The compiled ``_abscorr`` of the usability greedy returns numpy's ``|corr|`` and keeps its guards."""

import numpy as np

from mlframe.feature_selection.filters import _usability_greedy_steps as steps


def _reference(u: np.ndarray, v: np.ndarray) -> float:
    """The numpy form the kernel replaced: two standard deviations, then ``np.corrcoef``."""
    u, v = steps._f64(u), steps._f64(v)
    if u.size == 0 or float(np.std(u)) < 1e-12 or float(np.std(v)) < 1e-12:
        return 0.0
    r = np.corrcoef(u, v)[0, 1]
    return abs(float(r)) if np.isfinite(r) else 0.0


def test_matches_numpy_on_mixed_dtypes_and_heavy_tails():
    """Random pairs of both float widths, some with an exponential tail: the two forms agree to 1e-6 (the float32 standard deviations of the numpy form carry that much)."""
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(2, 4000))
        u = rng.standard_normal(n).astype(np.float32)
        v = (rng.random() * u + rng.random() * rng.standard_normal(n)).astype(np.float32 if trial % 2 else np.float64)
        if trial % 7 == 0:
            v = np.exp(5 * v)
        assert abs(steps._abscorr(u, v) - _reference(u, v)) < 1e-6


def test_guards_constant_nonfinite_and_empty():
    """A constant side, a NaN and an empty vector score 0.0; a common offset does not disturb a perfect correlation."""
    rng = np.random.default_rng(1)
    u = rng.standard_normal(500)
    assert steps._abscorr(u, np.full(500, 3.0)) == 0.0
    bad = u.copy()
    bad[3] = np.nan
    assert steps._abscorr(bad, u) == 0.0
    assert steps._abscorr(np.empty(0), np.empty(0)) == 0.0
    assert abs(steps._abscorr(u, 2.0 * u + 1e3) - 1.0) < 1e-6  # the operands are read at the fit's float width, so the offset stays within float32 resolution
