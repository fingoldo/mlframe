"""Regression sensor for A2#12 (S40): `_ratio_fit` empty-array `np.median` warning pitfall.

When ``base`` contains only non-finite or zero values, the prior implementation
called ``np.median([])`` -- a RuntimeWarning on NumPy 1.x and a hard error on
some NumPy 2.x configurations. The fix guards with ``base_finite.any()`` and
defaults ``scale=0.0``; ``eps`` then falls to the documented ``1e-12`` floor.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.training.composite.transforms.simple import _ratio_fit


def test_all_zero_base_does_not_raise_or_warn():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    base = np.zeros(4, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        params = _ratio_fit(y, base)
    assert params["eps"] == pytest.approx(1e-12)


def test_all_nonfinite_base_falls_to_eps_floor():
    y = np.array([1.0, 2.0, 3.0])
    base = np.array([np.nan, np.inf, -np.inf])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        params = _ratio_fit(y, base)
    assert params["eps"] == pytest.approx(1e-12)


def test_finite_mask_explicit_all_excluded():
    y = np.array([1.0, 2.0])
    base = np.array([10.0, 20.0])
    finite_mask_none = np.zeros(2, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        params = _ratio_fit(y, base, _finite_mask=finite_mask_none)
    assert params["eps"] == pytest.approx(1e-12)


def test_normal_base_unchanged():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    base = np.array([2.0, 4.0, 6.0, 8.0])
    params = _ratio_fit(y, base)
    # scale = median(|base|) = 5.0; eps = 5e-6
    assert params["eps"] == pytest.approx(5.0 * 1e-6)
