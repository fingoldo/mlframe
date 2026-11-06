"""The device CMI returns the host value when the candidate codes carry the -1 missing code.

``_native_kbins_quantile`` and the categorical encoders mark missing values as -1. The device CMI indexed its joint histograms with the
raw codes, so -1 wrote before the start of the table. For independent data the host gives about 8e-5 and the device gave 0.22.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_device_cmi_with_missing_code_matches_host():
    """Device and host CMI agree on candidate codes that contain -1."""
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe_gpu import _cmi_from_binned_cupy

    rng = np.random.default_rng(2)
    n = 4000
    x = rng.integers(-1, 4, n).astype(np.int64)
    y = rng.integers(0, 3, n).astype(np.int64)
    z = rng.integers(0, 3, n).astype(np.int64)

    host = float(_cmi_from_binned(x, y, z))
    device = float(_cmi_from_binned_cupy(x, y, z))

    assert device == pytest.approx(host, abs=1e-9)
