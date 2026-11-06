"""The device row max and min of the candidate columns propagate a NaN from any operand, as numpy does.

A NaN in a later operand used to be skipped by the comparison, giving the other operands' extremum (audit PARG1-3, wave 3).
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


def test_row_max_and_min_are_nan_when_a_later_operand_is_nan():
    """Operands [1, NaN, 3] give NaN for the row max and the row min on the device, as np.maximum and np.minimum do."""
    from mlframe.feature_selection.filters import _resident_candidate_mi as rc

    cols = [np.array([1.0, 2.0]), np.array([np.nan, 2.0]), np.array([3.0, 0.5])]
    out = cp.asnumpy(rc._build_best_existing_op_candidates_gpu([cp.asarray(c) for c in cols], cp))
    m = len(cols)
    npairs = m * (m - 1) // 2
    row_max = out[:, m + 4 * npairs]
    row_min = out[:, m + 4 * npairs + 1]
    assert np.isnan(row_max[0]) and np.isnan(row_min[0])
    assert row_max[1] == pytest.approx(2.0) and row_min[1] == pytest.approx(0.5)
