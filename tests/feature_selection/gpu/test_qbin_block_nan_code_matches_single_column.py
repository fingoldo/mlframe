"""A NaN row gets the same bin code from the block path (several columns) as from the single-column path.

The block count gave NaN code 0 while the per-column binner gives it the last code; the two must agree (audit PARG1-5, wave 3).
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


def test_nan_row_code_is_the_same_on_the_block_and_single_column_paths():
    """The NaN row of a column gets one code whether the column is binned alone or with another column."""
    from mlframe.feature_selection.filters import _qbin_block_gpu as qb

    rng = np.random.default_rng(0)
    base = rng.normal(size=400)
    base[7] = np.nan
    other = rng.normal(size=400)
    block = qb.resident_qbin_codes_block([base, other], 8, np.int64)
    single = qb.resident_qbin_codes_block([base], 8, np.int64)
    assert block is not None and single is not None
    assert int(cp.asnumpy(block[0])[7]) == int(cp.asnumpy(single[0])[7])
