"""The resident-operand cache stays under its device byte budget, keeping the newest entry, and still serves a hit by content."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_resident_operands as R


def _device_ok() -> bool:
    """Whether a CUDA device answers."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no driver: skipped
        return False


pytestmark = pytest.mark.skipif(not _device_ok(), reason="no CUDA device")


def test_cache_bytes_stay_under_the_budget(monkeypatch):
    """With a budget of three 8 KB operands, uploading six keeps at most three, the newest last."""
    R._FE_RESIDENT_OPERANDS.clear()
    monkeypatch.setattr(R, "_resident_bytes_budget", lambda _cp: 3 * 1000 * 8)
    arrays = [np.full(1000, float(i)) for i in range(6)]
    got = [R.resident_operand(a, ("t", i)) for i, a in enumerate(arrays)]
    total = sum(int(v.nbytes) for v in R._FE_RESIDENT_OPERANDS.values())
    assert total <= 3 * 1000 * 8 and len(R._FE_RESIDENT_OPERANDS) == 3
    assert next(reversed(R._FE_RESIDENT_OPERANDS.values())) is got[-1]
    np.testing.assert_array_equal(cp.asnumpy(got[-1]), arrays[-1])
    assert R.resident_operand(arrays[-1], ("t", 5)) is got[-1]
    R._FE_RESIDENT_OPERANDS.clear()
