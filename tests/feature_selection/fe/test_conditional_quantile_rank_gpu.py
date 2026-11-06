"""The device conditional quantile rank equals the host loop (ties, NaN / inf rows, a negative bin id, a single bin)."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _conditional_quantile_rank_fe as cqr

cp = pytest.importorskip("cupy")


def _host(x, bins, monkeypatch):
    """The host loop's result (device path forced off)."""
    monkeypatch.setenv("MLFRAME_CQR_DEVICE", "0")
    return cqr.conditional_quantile_rank_fe(x, bins)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_device_matches_host(seed, monkeypatch):
    """Ties, NaN / inf rows and negative bin ids give the host loop's numbers."""
    rng = np.random.default_rng(seed)
    n = 5000
    x = np.round(rng.normal(size=n), 1)  # heavy ties
    x[rng.integers(0, n, 40)] = np.nan
    x[rng.integers(0, n, 10)] = np.inf
    bins = rng.integers(-2, 7, n)
    host = _host(x, bins, monkeypatch)
    from mlframe.feature_selection.filters._conditional_quantile_rank_gpu import conditional_quantile_rank_device

    dev = conditional_quantile_rank_device(x.astype(np.float32), bins)
    assert dev is not None
    ref = _host(x.astype(np.float32), bins, monkeypatch)
    np.testing.assert_array_equal(np.isnan(dev), np.isnan(ref))
    np.testing.assert_allclose(dev[~np.isnan(dev)], ref[~np.isnan(ref)], rtol=0, atol=0)
    assert host.shape == dev.shape


def test_single_bin_and_dispatch(monkeypatch):
    """One bin through the public function with the device path forced on equals the host."""
    x = np.arange(1000, dtype=np.float32)
    bins = np.zeros(1000, dtype=np.int64)
    ref = _host(x, bins, monkeypatch)
    monkeypatch.setenv("MLFRAME_CQR_DEVICE", "1")
    got = cqr.conditional_quantile_rank_fe(x, bins)
    np.testing.assert_array_equal(got, ref)
