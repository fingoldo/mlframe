"""The device rounding and digit columns equal the host expressions exactly, and the generators give the same frames with the device forced on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.gpu._gpu_support import has_device

from mlframe.feature_selection.filters import _numeric_decompose_gpu as ndg
from mlframe.feature_selection.filters import _numeric_decompose_fe as nd


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")


def _cols():
    """Columns with ties at half-integers, negatives, NaN and inf."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(5000) * 50
    a[:6] = [0.5, 1.5, 2.5, -0.5, -1.5, 0.25]
    a[10::97] = np.nan
    a[11::131] = np.inf
    return [a, rng.integers(-300, 300, 5000) / 8.0]


def test_device_columns_equal_host_exactly():
    """Every rounding and digit output is bit-equal to the host function, NaN positions included."""
    _needs_gpu()
    cols = _cols()
    kinds, params = (0, 0, 0, 1, 1, 1), (1, 0.1, 0.01, 0, 1, 2)
    host = ndg.decompose_columns_host(cols, kinds, params)
    dev = ndg.decompose_columns_gpu(cols, kinds, params)
    np.testing.assert_array_equal(dev[:, :3].astype(np.float32), host[:, :3].astype(np.float32))
    np.testing.assert_array_equal(dev[:, 3:], host[:, 3:])


def test_generators_match_with_the_device_forced(monkeypatch):
    """generate_rounding_features and generate_digit_features return identical frames on either backend."""
    _needs_gpu()
    X = pd.DataFrame({"a": _cols()[0], "b": _cols()[1]})
    monkeypatch.setattr(ndg, "decompose_device_chosen", lambda n, p, o: False)
    r0, d0 = nd.generate_rounding_features(X), nd.generate_digit_features(X)
    monkeypatch.setattr(ndg, "decompose_device_chosen", lambda n, p, o: True)
    r1, d1 = nd.generate_rounding_features(X), nd.generate_digit_features(X)
    pd.testing.assert_frame_equal(r1, r0)
    pd.testing.assert_frame_equal(d1, d0)
