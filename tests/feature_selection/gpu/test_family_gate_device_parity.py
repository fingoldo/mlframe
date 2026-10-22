"""The device-resident family gate decides exactly as the host family gate on the same codes, and never copies the codes back."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover - no device
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._family_maxt_gate import family_pass_batches
from mlframe.feature_selection.filters._family_maxt_gate_device import family_pass_batches_device
from mlframe.feature_selection.filters._lazy_host_codes import LazyHostCodes


def _family(seed, n=6000, signal=True):
    """Three batches of binned columns: a clear signal batch, a pure-noise batch and a mixed batch, with a 4-class target."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    noise = lambda k: rng.integers(0, 8, (n, k))
    sig = np.where(rng.random((n, 3)) < 0.5, y[:, None] * 2, rng.integers(0, 8, (n, 3)))
    batches = [sig if signal else noise(3), noise(5), np.column_stack([noise(2), np.where(rng.random(n) < 0.3, y * 2, rng.integers(0, 8, n))])]
    return [b.astype(np.int8) for b in batches], y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_device_flags_equal_host_flags(seed):
    """Per-batch pass flags agree between the device gate and the host gate."""
    batches, y = _family(seed)
    host = family_pass_batches([np.ascontiguousarray(b) for b in batches], y, alpha=0.01, n_perm=999)
    lazy = [LazyHostCodes(cp.asarray(b), np.int8) for b in batches]
    dev = family_pass_batches_device(lazy, y, 0.01, 999, 0)
    assert dev == host
    assert host[0] and not host[1], host


def test_the_device_gate_copies_back_no_codes_and_declines_host_batches():
    """Scoring resident batches does not materialise their host copy; plain host batches make the device path decline (the host gate answers)."""
    batches, y = _family(3)
    lazy = [LazyHostCodes(cp.asarray(b), np.int8) for b in batches]
    assert family_pass_batches_device(lazy, y, 0.01, 999, 0) is not None
    assert all(lz._host is None for lz in lazy)
    assert family_pass_batches_device(batches, y, 0.01, 999, 0) is None
