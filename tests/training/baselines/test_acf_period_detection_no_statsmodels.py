"""The dummy-baselines ACF period detector must not depend on statsmodels.

Its statsmodels import broke under pandas 3 with a TypeError (``deprecate_kwarg() missing ... 'new_arg_name'``)
that the ``except ImportError`` guard did not catch, crashing time-series baseline detection. It now uses mlframe's
numpy FFT ACF, which equals statsmodels' default estimator; these pins check the equality and that detection works
with statsmodels made unimportable.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from mlframe.reporting.charts._acf import acf_fft
from mlframe.training.baselines._dummy_timeseries import _detect_acf_periods


def _seasonal(n=3000, period=7, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10.0 * np.sin(2 * np.pi * t / period) + rng.normal(0, 1.0, n)


def test_acf_fft_equals_statsmodels_default():
    """mlframe's ACF equals statsmodels acf(fft=True) (demeaned, biased 1/n)."""
    try:
        from statsmodels.tsa.stattools import acf
    except Exception as e:  # any import failure (incl. version clashes) -> nothing to compare against
        pytest.skip(f"statsmodels not importable: {e}")
    y = np.diff(_seasonal())
    ours, _ = acf_fft(y, nlags=40)
    theirs = acf(y, nlags=40, fft=True)
    np.testing.assert_allclose(ours, theirs[1:], rtol=0, atol=1e-10)


def test_period_detection_works_without_statsmodels(monkeypatch):
    """A broken/missing statsmodels must not affect period detection."""
    monkeypatch.setitem(sys.modules, "statsmodels", None)  # any `import statsmodels...` now raises
    monkeypatch.setitem(sys.modules, "statsmodels.tsa", None)
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", None)
    y = _seasonal(period=7)
    periods = _detect_acf_periods(y, n_train=len(y))
    assert 7 in periods, periods
