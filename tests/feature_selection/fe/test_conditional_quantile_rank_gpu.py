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


def test_generate_recipes_equal_between_paths(monkeypatch):
    """The generated columns and the stored per-bin sorted references are identical with the device path on and off."""
    import pandas as pd

    rng = np.random.default_rng(3)
    X = pd.DataFrame({f"f{i}": np.round(rng.normal(size=6000), 2).astype(np.float32) for i in range(4)})
    X.iloc[::97, 0] = np.nan
    monkeypatch.setenv("MLFRAME_CQR_DEVICE", "0")
    enc0, rec0 = cqr.generate_conditional_quantile_rank_features(X, list(X.columns))
    monkeypatch.setenv("MLFRAME_CQR_DEVICE", "1")
    enc1, rec1 = cqr.generate_conditional_quantile_rank_features(X, list(X.columns))
    assert list(enc0.columns) == list(enc1.columns)
    np.testing.assert_array_equal(enc0.to_numpy(), enc1.to_numpy())
    for name, r0 in rec0.items():
        r1 = rec1[name]
        assert len(r0["sorted_bins"]) == len(r1["sorted_bins"])
        for a, b in zip(r0["sorted_bins"], r1["sorted_bins"]):
            np.testing.assert_array_equal(a, b)
