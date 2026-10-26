"""With the project-wide GPU switch off the family null never touches the device, whatever the tuned backend choice or STRICT say."""
import numpy as np
import pytest

from mlframe.feature_selection.filters import _family_maxt_gate as gate


def _inputs():
    """A small family: eight 6-bin columns and a binary label."""
    rng = np.random.default_rng(0)
    cols = rng.integers(0, 6, (8, 600)).astype(np.int64)
    n_x = np.full(8, 6, dtype=np.int64)
    y = rng.integers(0, 2, 600).astype(np.int64)
    return cols, n_x, y


@pytest.mark.parametrize("strict", ["1", "0", ""])
def test_gpu_disable_keeps_the_null_on_the_host(monkeypatch, strict):
    """A tuned choice of "device" and an explicit STRICT=1 are both overridden by MLFRAME_DISABLE_GPU=1: the device bit sets are never even requested and the host result is returned."""
    from mlframe.feature_selection.filters import _family_maxt_cuda_bits as bits

    attempts = []

    def _record(*_a, **_k):
        """Count the attempt; the gate swallows the failure and counts on the host, so the count is what the test reads."""
        attempts.append(1)
        raise RuntimeError("no device in this test")

    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    if strict:
        monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", strict)
    monkeypatch.setattr(gate, "_family_null_backend", lambda *a, **k: "device")
    monkeypatch.setattr(bits, "DeviceBitsets", _record)
    monkeypatch.setattr(bits, "null_max_bits_device", _record)
    cols, n_x, y = _inputs()
    out = gate.family_null_max(cols, n_x, y, 2, 20, np.random.default_rng(1))
    assert out.shape == (20,) and np.isfinite(out).all()
    assert attempts == [], "the device path was requested with the GPU switched off"
