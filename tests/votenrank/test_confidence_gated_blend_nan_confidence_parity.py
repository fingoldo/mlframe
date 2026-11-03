"""NaN confidence takes the default weight in every backend of the confidence-gated blend."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

cgb = importlib.import_module("mlframe.votenrank.confidence_gated_blend")


@pytest.mark.parametrize("kernel_name", ["_blend_njit", "_blend_njit_parallel"])
def test_nan_confidence_matches_numpy_reference(kernel_name):
    """The compiled kernels agree with the numpy path on NaN and inf confidences, including NaN in the blended inputs."""
    rng = np.random.default_rng(0)
    n = 5000
    ens, aux = rng.random(n), rng.random(n)
    conf = rng.random(n)
    conf[::7] = np.nan
    conf[3::11] = np.inf
    conf[5::13] = -np.inf
    ens[10] = np.nan
    expected = cgb._blend_numpy(ens, aux, conf, 0.5, 0.8, 0.2)
    got = getattr(cgb, kernel_name)(ens, aux, conf, 0.5, 0.8, 0.2)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got, expected, equal_nan=True)
