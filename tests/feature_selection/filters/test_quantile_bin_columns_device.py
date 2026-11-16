"""``quantile_bin_columns`` gives the same codes from the device twin of the host binner (ties, few levels, two-class, constant and rare columns included)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import quantile_bin_columns


def _matrix(n=6000):
    """Float32 columns of every kind the binner has a rule for."""
    rng = np.random.default_rng(0)
    cols = [rng.normal(size=n), np.round(rng.normal(size=n), 1), rng.integers(0, 3, n), rng.random(n) < 0.3, np.ones(n), rng.integers(0, 2, n),
            np.where(rng.random(n) < 0.95, 0.0, rng.normal(size=n)), rng.integers(0, 12, n), rng.random(n) < 0.001, rng.normal(size=n)]
    return np.ascontiguousarray(np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1))


@pytest.mark.parametrize("nbins", [2, 10, 20])
def test_device_codes_equal_the_host_codes(monkeypatch, nbins):
    """Every column's codes are identical on either side of ``MLFRAME_QBIN_COLUMNS_DEVICE``."""
    pytest.importorskip("cupy")
    X = _matrix()
    monkeypatch.setenv("MLFRAME_QBIN_COLUMNS_DEVICE", "0")
    host = quantile_bin_columns(X, nbins, dtype=np.int16)
    monkeypatch.setenv("MLFRAME_QBIN_COLUMNS_DEVICE", "1")
    dev = quantile_bin_columns(X, nbins, dtype=np.int16)
    for j, (h, d) in enumerate(zip(host, dev)):
        assert np.array_equal(h, d), j
