"""The GPU rank binning keeps distinct values apart when a column's offset dwarfs its spread.

Values 1e8 + k * 1e-4 are distinct, but float32 merges them into one bin, so the MI of a split at the midpoint read zero instead of ln 2 (audit
PARG3-1, wave 3). Ordinary columns keep the float32 criterion dtype.
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


def test_offset_column_split_keeps_its_mutual_information():
    """A column of distinct values near 1e8 that splits the target in halves has MI close to ln 2 on the device."""
    from mlframe.feature_selection.filters import _hermite_fe_mi as hm

    n = 2000
    x = 1e8 + np.arange(n) * 1e-4
    y = (np.arange(n) >= n // 2).astype(np.int64)
    mi = np.asarray(hm._plugin_mi_classif_batch_cuda(x.reshape(-1, 1), y, 4))
    assert mi[0] == pytest.approx(np.log(2), abs=0.05)


def test_ordinary_columns_keep_the_float32_criterion_dtype():
    """A standard-normal column is not offset-dominated and stays on the criterion dtype."""
    from mlframe.feature_selection.filters import _hermite_fe_mi as hm
    from mlframe.feature_selection.filters._fe_gpu_batch._devices import crit_float_dtype

    x = np.random.default_rng(0).normal(size=(500, 2))
    assert hm._binning_dtype(x) == crit_float_dtype()
