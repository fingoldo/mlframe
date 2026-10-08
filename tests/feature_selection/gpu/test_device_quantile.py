"""The device quantile reproduces ``np.quantile`` (linear) bit for bit, including ties, and defers NaN input to the host."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._device_quantile import device_quantile


@pytest.mark.parametrize("n", [1, 2, 7, 1000, 123_457, 1_000_000])
@pytest.mark.parametrize("kind", ["normal", "ties", "heavy", "binary"])
def test_matches_numpy_exactly(n, kind):
    """Exact equality over several shapes of data and quantile grids."""
    rng = np.random.default_rng(n % 97)
    x = {"normal": rng.standard_normal(n), "ties": np.round(rng.standard_normal(n), 1), "heavy": rng.standard_normal(n) ** 3, "binary": (rng.random(n) > 0.4).astype(np.float64)}[kind]
    for qs in (np.linspace(0.0, 1.0, 11)[1:-1], np.linspace(0.0, 1.0, 21)[1:-1], np.array([0.0, 0.5, 1.0]), np.array([0.01, 0.999])):
        got = device_quantile(x, qs)
        assert got is not None
        assert np.array_equal(got, np.quantile(x, qs))


def test_nan_input_is_left_to_the_host():
    """A NaN makes np.quantile propagate NaN; the device path declines so the host decides."""
    x = np.random.default_rng(0).standard_normal(500)
    x[10] = np.nan
    assert device_quantile(x, np.array([0.1, 0.5])) is None
