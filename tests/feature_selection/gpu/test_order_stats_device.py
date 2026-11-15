"""The device per-category order statistics equal the host sort and sweep; per_category_order_stats returns the same dict with the device forced, and the device path demonstrably ran."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _order_stats_gpu as g
from mlframe.feature_selection.filters import _target_encoding_order_stats as oso


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")


def _data(n=30_000, cats=300):
    """Skewed y over categories of very different sizes, including an empty one, a single-row one and ties."""
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(cats) * 0.3)
    p[7] = 0.0
    p /= p.sum()
    inv = rng.choice(cats, size=n, p=p).astype(np.int64)
    inv[0] = 5
    y = np.round(rng.standard_exponential(n) + 0.05 * inv, 1)
    return inv, y, cats


def test_device_equals_host_for_every_category_and_statistic():
    """All seven raw arrays agree with the numba sweep to 1e-12 relative, NaNs (empty categories, stats not asked) in the same places."""
    _needs_gpu()
    inv, y, cats = _data()
    counts = np.bincount(inv, minlength=cats).astype(np.int64)
    host = g._sweep_host(inv, y, counts)
    dev = g._sweep_device(inv, y, counts)
    assert np.array_equal(np.isnan(host), np.isnan(dev))
    np.testing.assert_allclose(dev, host, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_per_category_order_stats_identical_with_the_device_forced_and_it_ran(monkeypatch):
    """The encoder's output dict is the same with the device off and on (fallback to the global value for rare cells included); the device scorer is called once."""
    _needs_gpu()
    inv, y, cats = _data()
    stats = list(oso.ORDER_STATS)
    gstats = oso.global_order_stats(y, stats)
    monkeypatch.setattr(g, "order_device_chosen", lambda n: False)
    want = oso.per_category_order_stats(inv, y, cats, stats, gstats)
    monkeypatch.setattr(g, "order_device_chosen", lambda n: True)
    calls = {"n": 0}
    real = g.segment_order_stats_gpu

    def counting(*a, **k):
        """The device scorer, counting calls so the test proves the device path ran."""
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(g, "segment_order_stats_gpu", counting)
    got = oso.per_category_order_stats(inv, y, cats, stats, gstats)
    assert calls["n"] == 1
    assert set(got) == set(want)
    for stat in want:
        np.testing.assert_allclose(got[stat], want[stat], rtol=1e-12, atol=1e-12)
