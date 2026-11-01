"""The routing correlation helper equals abs(np.corrcoef) and the routed basis is unchanged."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _abs_pearson


def test_equals_corrcoef_in_both_precisions():
    """Float32 and float64 candidate values, positive and negative association."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    yc = y - y.mean()
    yn = float(np.sqrt(yc @ yc))
    for dt in (np.float32, np.float64):
        for sign in (1.0, -1.0):
            v = (sign * 0.4 * y + rng.normal(size=5000)).astype(dt)
            want = abs(float(np.corrcoef(v, y.astype(dt))[0, 1]))
            np.testing.assert_allclose(_abs_pearson(v, yc, yn), want, rtol=1e-5)
