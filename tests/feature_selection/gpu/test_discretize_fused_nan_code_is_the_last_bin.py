"""The fused resident discretiser gives a NaN row the last bin, as its own fallback and the CPU binner do.

The fused kernels put a NaN row in bin 0 (audit PARG3-3, wave 3).
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    _HAS_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def test_nan_row_gets_the_last_bin_on_the_fused_path():
    """Column [1..10, NaN] with four bins: the NaN row codes to bin 3, as the CPU binner gives."""
    from mlframe.feature_selection.filters import _gpu_resident_discretize as gd

    col = np.array([float(i) for i in range(1, 11)] + [np.nan]).reshape(-1, 1)
    codes = cp.asnumpy(gd._gpu_resident_discretize_codes(cp.asarray(col), 4))
    assert int(codes.reshape(-1)[-1]) == 3
