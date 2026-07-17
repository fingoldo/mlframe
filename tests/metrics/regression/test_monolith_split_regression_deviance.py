"""Sensor: ``_regression_extras.py`` deviance carve into ``_regression_deviance.py``.

Verifies parent + ``metrics.core`` re-export identity AND calls into the moved
kernels (forcing the njit compile + cross-ref resolution), with sklearn parity.
"""

from __future__ import annotations

import numpy as np


def test_deviance_reexport_identity():
    from mlframe.metrics.regression import _regression_deviance as sib
    from mlframe.metrics.regression import _regression_extras as parent
    from mlframe.metrics import core

    for nm in ("fast_poisson_deviance", "fast_gamma_deviance", "fast_tweedie_deviance"):
        assert getattr(parent, nm) is getattr(sib, nm)
        assert getattr(core, nm) is getattr(sib, nm)


def test_deviance_bodies_callable_with_sklearn_parity():
    from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance

    from mlframe.metrics.regression._regression_deviance import (
        fast_gamma_deviance,
        fast_poisson_deviance,
        fast_tweedie_deviance,
    )

    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.1, 1.9, 3.2, 3.8])
    assert abs(fast_poisson_deviance(yt, yp) - mean_poisson_deviance(yt, yp)) < 1e-9
    assert abs(fast_gamma_deviance(yt, yp) - mean_gamma_deviance(yt, yp)) < 1e-9
    assert abs(fast_tweedie_deviance(yt, yp, power=1.5) - mean_tweedie_deviance(yt, yp, power=1.5)) < 1e-9
    # power=0 short-circuits to MSE
    assert abs(fast_tweedie_deviance(yt, yp, power=0.0) - np.mean((yt - yp) ** 2)) < 1e-12


def test_deviance_invalid_rows_return_nan():
    from mlframe.metrics.regression._regression_deviance import fast_poisson_deviance

    # all rows invalid (y_pred <= 0) -> NaN
    assert np.isnan(fast_poisson_deviance(np.array([1.0, 2.0]), np.array([-1.0, 0.0])))


def test_general_tweedie_pow_reuse_matches_sklearn_tightly():
    """General-kernel pow-reuse (yp**(2-p) == yp**(1-p)*yp) must stay bit-tight to sklearn.

    Pins the optimization that drops one variable-exponent pow per row; a regression to a
    separate yp**(2-p) call would not move this assertion, but any algebraic mistake would.
    Covers 1<p<2 (compound Poisson-Gamma), p>2, p>3 and the y_true==0 support branch.
    """
    from sklearn.metrics import mean_tweedie_deviance
    from mlframe.metrics.regression._regression_deviance import fast_tweedie_deviance

    rng = np.random.default_rng(7)
    yt = rng.gamma(2.0, 1.5, 4000)
    yp = rng.gamma(2.0, 1.5, 4000) + 0.05
    for power in (1.3, 1.5, 1.9, 2.5, 3.5):
        got = fast_tweedie_deviance(yt, yp, power=power)
        exp = mean_tweedie_deviance(yt, yp, power=power)
        assert abs(got - exp) < 1e-10, f"power={power}: {got} vs sklearn {exp}"

    # y_true == 0 rows take the term_y=0 support branch (valid only for 1<p<2 in sklearn).
    yt0 = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
    yp0 = np.array([0.5, 1.2, 1.8, 2.0, 2.5])
    got = fast_tweedie_deviance(yt0, yp0, power=1.5)
    exp = mean_tweedie_deviance(yt0, yp0, power=1.5)
    assert abs(got - exp) < 1e-10
