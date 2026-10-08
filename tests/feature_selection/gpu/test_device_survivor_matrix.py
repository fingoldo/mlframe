"""A device-backed survivor matrix is binned and edge-fitted on the device with the same results as the host formulation."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._lazy_host_codes import LazyHostCodes
from mlframe.feature_selection.filters._mrmr_fe_step._step_score_parts2 import _device_codes_to_host
from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch
from mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary import _device_fit_edges
from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit


def _matrix(n: int, k: int, seed: int) -> np.ndarray:
    """Heavy-tailed, tie-heavy and constant-ish float64 columns."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, k)) ** 3
    m[:, 0] = np.round(m[:, 0])
    m[:, 1] = 0.0
    return m


@pytest.mark.parametrize("n, k, nbins", [(2000, 4, 10), (999, 3, 7)])
def test_device_codes_equal_the_host_batch_discretiser(n, k, nbins):
    """Binning on the device gives the same codes as ``discretize_2d_quantile_batch`` and copies back only the int codes."""
    m = _matrix(n, k, 0)
    lazy = LazyHostCodes(cp.asarray(m), np.float64)
    with residency_audit() as rep:
        dev_codes = _device_codes_to_host(lazy, nbins, np.int8)
    assert dev_codes is not None
    np.testing.assert_array_equal(dev_codes, discretize_2d_quantile_batch(m, n_bins=nbins, dtype=np.int8))
    assert sum(rep.d2h) <= dev_codes.nbytes + 1024  # the codes, never the float matrix


def test_a_host_matrix_is_left_to_the_host_path():
    """No device backing -> ``None`` so the caller bins on the host as before."""
    assert _device_codes_to_host(np.zeros((10, 2)), 10, np.int8) is None


@pytest.mark.parametrize("method", ["quantile", "uniform"])
def test_device_fit_edges_equal_the_host_edges(method):
    """The persisted recipe edges computed on the device equal the host ``nanpercentile`` / ``linspace`` edges, and only the edges cross to the host."""
    col = _matrix(5000, 3, 1)[:, 2]
    lazy = LazyHostCodes(cp.asarray(col), np.float64)
    with residency_audit() as rep:
        dev_edges = _device_fit_edges(lazy, method, 10)
    assert not rep.bulk_d2h
    if method == "quantile":
        host = np.nanpercentile(col, np.linspace(0.0, 100.0, 11))
    else:
        host = np.linspace(col.min(), col.max(), 11)
    np.testing.assert_allclose(dev_edges, host, rtol=1e-12, atol=0)
    assert _device_fit_edges(col, method, 10) is None


def test_a_column_of_a_lazy_matrix_stays_lazy_and_equals_the_host_column():
    """``matrix[:, j]`` is a device view; reading it later gives exactly the host column."""
    m = _matrix(300, 3, 2)
    lazy = LazyHostCodes(cp.asarray(m), np.float64)
    with residency_audit() as rep:
        col = lazy[:, 1]
        assert isinstance(col, LazyHostCodes) and col.shape == (300,)
    assert not rep.d2h
    np.testing.assert_array_equal(np.asarray(col), m[:, 1])
