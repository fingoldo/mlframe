"""Sensor: ``_regression_extras.py`` deviance carve into ``_regression_deviance.py``.

Verifies parent + ``metrics.core`` re-export identity AND calls into the moved
kernels (forcing the njit compile + cross-ref resolution), with sklearn parity.
"""
from __future__ import annotations

import numpy as np


def test_deviance_reexport_identity():
    from mlframe.metrics import _regression_deviance as sib
    from mlframe.metrics import _regression_extras as parent
    from mlframe.metrics import core

    for nm in ("fast_poisson_deviance", "fast_gamma_deviance", "fast_tweedie_deviance"):
        assert getattr(parent, nm) is getattr(sib, nm)
        assert getattr(core, nm) is getattr(sib, nm)


def test_deviance_bodies_callable_with_sklearn_parity():
    from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance

    from mlframe.metrics._regression_deviance import (
        fast_gamma_deviance, fast_poisson_deviance, fast_tweedie_deviance,
    )

    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.1, 1.9, 3.2, 3.8])
    assert abs(fast_poisson_deviance(yt, yp) - mean_poisson_deviance(yt, yp)) < 1e-9
    assert abs(fast_gamma_deviance(yt, yp) - mean_gamma_deviance(yt, yp)) < 1e-9
    assert abs(fast_tweedie_deviance(yt, yp, power=1.5) - mean_tweedie_deviance(yt, yp, power=1.5)) < 1e-9
    # power=0 short-circuits to MSE
    assert abs(fast_tweedie_deviance(yt, yp, power=0.0) - np.mean((yt - yp) ** 2)) < 1e-12


def test_deviance_invalid_rows_return_nan():
    from mlframe.metrics._regression_deviance import fast_poisson_deviance

    # all rows invalid (y_pred <= 0) -> NaN
    assert np.isnan(fast_poisson_deviance(np.array([1.0, 2.0]), np.array([-1.0, 0.0])))
