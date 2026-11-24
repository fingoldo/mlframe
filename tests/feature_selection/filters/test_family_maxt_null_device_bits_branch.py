"""The device bit-set branch of the family null gives the same maxima as the host path for the same generator state.

Strict GPU mode routes the null through ``_family_null_max_device_bits``; the test checks that the branch ran (it is wrapped to record the call) and that its maxima equal the host
run's. Skips without cupy or a usable device.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_selection.filters._family_maxt_gate as gate

cp = pytest.importorskip("cupy")

try:  # audit-exempt: except_skip_masks_call_under_test -- optional CUDA device guard: the probe decides whether the module can run, it is not the code under test
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - no CUDA runtime on this host
    pytest.skip("no usable CUDA device", allow_module_level=True)


def _inputs(seed: int = 9):
    """Four candidate columns of four bins on 600 rows; the first two carry signal about a three-class label."""
    rng = np.random.default_rng(seed)
    n = 600
    y = rng.integers(0, 3, n).astype(np.int64)
    cols = rng.integers(0, 4, (4, n)).astype(np.int64)
    cols[0] = (y + rng.integers(0, 2, n)) % 4
    cols[1] = (2 * y + rng.integers(0, 2, n)) % 4
    n_x = np.full(4, 4, dtype=np.int64)
    return cols, n_x, y


def test_device_bit_branch_matches_the_host_maxima(monkeypatch):
    """Under strict GPU mode the device branch runs and returns the host run's maxima, block by block."""
    cols, n_x, y = _inputs()
    n_perm = 24

    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    host = gate.family_null_max(cols, n_x, y, 3, n_perm, np.random.default_rng(21), block=8)

    calls: list[int] = []
    real = gate._family_null_max_device_bits

    def recording(*args, **kwargs):
        """Record that the device branch ran, then call the real function."""
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setattr(gate, "_family_null_max_device_bits", recording)
    device = gate.family_null_max(cols, n_x, y, 3, n_perm, np.random.default_rng(21), block=8)

    assert calls, "the device bit branch was not taken under strict GPU mode"
    np.testing.assert_allclose(np.asarray(device), np.asarray(host), rtol=1e-12, atol=1e-12)
