"""The batched leg selection of many columns returns each column's own device selection, with two MI calls per block of columns."""

import numpy as np
from tests.feature_selection.gpu._gpu_support import require_device
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_batched_mi, _wavelet_basis_fe_batched as batched


def _columns(n: int = 6000) -> "tuple[list, np.ndarray, list, list]":
    """Six columns (one with a localised effect on the target, one constant) with their ``lo`` / ``span``, and a binary target."""
    rng = np.random.default_rng(0)
    xs = [rng.random(n).astype(np.float32) for _ in range(5)] + [np.full(n, 0.5, dtype=np.float32)]
    y = ((xs[1] > 0.25) & (xs[1] < 0.5)).astype(np.int64) ^ (rng.random(n) < 0.05)
    los = [float(x.min()) for x in xs]
    spans = [max(float(x.max()) - lo, 1e-12) for x, lo in zip(xs, los)]
    return xs, y.astype(np.int64), los, spans


def test_batch_equals_the_per_column_selection(monkeypatch):
    """Every column gets the legs its own call gets (the informative one a non-empty list, the constant one none), from 2 MI calls for the whole batch."""
    require_device()
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    xs, y, los, spans = _columns()
    kw = dict(max_scale=3, max_legs=4, scale_sigma=3.0)
    want = [batched._select_wavelet_legs_batched_device(x, y, lo, sp, **kw) for x, lo, sp in zip(xs, los, spans)]
    assert all(w is not None for w in want), "the device leg selection returned nothing on a host with a usable device"
    calls = []
    real = _fe_batched_mi.binned_mi_from_codes_gpu

    def spy(codes, *args, **kwargs):
        """Record the width of each MI call."""
        calls.append(int(codes.shape[1]))
        return real(codes, *args, **kwargs)

    monkeypatch.setattr(_fe_batched_mi, "binned_mi_from_codes_gpu", spy)
    got = batched.select_wavelet_legs_columns_device(xs, y, los, spans, **kw)
    assert got is not None, "the batched route returned nothing on a host with a usable device"
    assert got == want
    assert len(got[1]) > 0 and got[5] == []
    assert len(calls) == 2 and calls[0] == calls[1] > 15


def test_small_block_budget_gives_the_same_legs(monkeypatch):
    """A budget of two columns per block scores in several blocks and admits the same legs."""
    require_device()
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    xs, y, los, spans = _columns()
    kw = dict(max_scale=3, max_legs=4, scale_sigma=3.0)
    whole = batched.select_wavelet_legs_columns_device(xs, y, los, spans, **kw)
    assert whole is not None, "the batched route returned nothing on a host with a usable device"
    monkeypatch.setattr(batched, "_DEVICE_LEG_BLOCK_BYTES", 2 * 2 * 15 * 8 * len(y))  # two columns, each held twice while its block is scored
    assert batched.select_wavelet_legs_columns_device(xs, y, los, spans, **kw) == whole
