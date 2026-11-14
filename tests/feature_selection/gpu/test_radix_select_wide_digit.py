"""The opt-in 13-bit-first-digit radix select (v6) returns the order statistics of v5 bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_select_f32_wide as wide
from mlframe.feature_selection.filters._gpu_resident_select_f32_compact import _get_radix_select_f32_v5_kernel


def _run(kern, data, ranks, shared):
    """Order statistics (R, K) of the (K, n) ``data`` at ``ranks`` from one select kernel."""
    K, n = data.shape
    out = cp.zeros((len(ranks), K), dtype=cp.float32)
    cap = max(1024, n // 4)
    kern(
        (K,),
        (256,),
        (data, np.int64(n), np.int32(K), cp.asarray(ranks), np.int32(len(ranks)), out, cp.empty((K, cap), dtype=cp.uint32), np.int64(cap)),
        shared_mem=max(len(ranks) * 1024, shared),
    )
    return cp.asnumpy(out)


@pytest.mark.parametrize("kind", ["normal", "ties", "lognormal"])
def test_v6_equals_v5(monkeypatch, kind):
    """Normal, tied and heavy-tailed columns, ranks at the decile positions: the same order statistics."""
    monkeypatch.setenv("MLFRAME_RADIX_F32_V6", "1")
    rng = np.random.default_rng(5)
    n, K = 20000, 6
    x = rng.standard_normal((K, n))
    x = {"normal": x, "ties": np.round(x * 3), "lognormal": np.exp(x)}[kind].astype(np.float32)
    q = np.linspace(0, 1, 11)[1:-1] * (n - 1)
    ranks = np.unique(np.concatenate([np.floor(q), np.minimum(np.floor(q) + 1, n - 1)])).astype(np.int64)
    kern6 = wide._get_radix_select_f32_v6_kernel()
    assert kern6 is not None
    d = cp.asarray(x)
    assert np.array_equal(_run(kern6, d, ranks, wide.FIRST_DIGIT_SHARED_BYTES), _run(_get_radix_select_f32_v5_kernel(), d, ranks, 0))


def test_v6_is_off_by_default(monkeypatch):
    """Rejected as a default: the getter returns None unless the switch is set."""
    monkeypatch.delenv("MLFRAME_RADIX_F32_V6", raising=False)
    assert wide._get_radix_select_f32_v6_kernel() is None
