"""The device G statistics of the added-information gate give the host p-values, and oversize or sparse candidates keep the host decisions."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _added_information_gpu as g
from mlframe.feature_selection.filters._added_information_gate import added_information_p


def _needs_gpu():
    """Skip when no usable CUDA device is visible."""
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no usable CUDA device")


def _data(n=20_000):
    """A source of 4 levels, a binary target, and candidates: a copy of the source, noise, a column carrying what the source lacks, and a wide one."""
    rng = np.random.default_rng(4)
    s = rng.integers(0, 4, n)
    extra = rng.integers(0, 10, n)
    t = ((s + extra) % 2).astype(np.int64)
    cands = [s.copy(), rng.integers(0, 10, n), extra, rng.integers(0, 600, n)]
    return cands, s, t


def test_p_values_equal_the_host(monkeypatch):
    """Forcing the device on, every candidate's p-value equals added_information_p (the wide one has a table over the limit and goes to the host)."""
    _needs_gpu()
    cands, s, t = _data()
    monkeypatch.setattr(g, "added_device_chosen", lambda n, k: True)
    got = g.added_information_p_many(cands, s, t)
    want = [added_information_p(c, s, t) for c in cands]
    for a, b in zip(got, want):
        if np.isnan(b):
            assert np.isnan(a)
        else:
            assert a == pytest.approx(b, rel=1e-9, abs=1e-12)


def test_g_statistics_match_the_host_expression():
    """The kernel's G equals the numpy expression for candidates of different level counts."""
    _needs_gpu()
    cands, s, t = _data()
    from mlframe.feature_selection.filters._added_information_gate import _dense_codes

    dense = [_dense_codes(c) for c in cands[:3]]
    mat = np.stack([d[0] for d in dense])
    bcs = np.array([d[1] for d in dense])
    dev = g.g_statistics_gpu(mat, bcs, s, 4, t, 2)
    # candidates of equal level count share the host helper; compare each by itself
    for j in range(3):
        host = g._g_host(mat[j : j + 1], bcs[j : j + 1], s, 4, t, 2)[0]
        assert dev[j] == pytest.approx(host, rel=1e-10, abs=1e-9)
