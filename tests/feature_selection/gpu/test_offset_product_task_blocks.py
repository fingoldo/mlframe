"""The offset-product device scan in blocks of tasks (each uploading only its columns) equals the single-launch scan bit for bit and the numba scan to rounding."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection._benchmarks.kernel_tuning_cache import _auto_tune_sweeps_c as sweeps
from mlframe.feature_selection.filters import _device_blocks as db
from mlframe.feature_selection.filters._offset_product_gpu import scan_offset_products_gpu
from mlframe.feature_selection.filters._offset_product_kernels import N_BASELINES, scan_offset_products


def test_task_blocks_equal_single_launch_and_numba(monkeypatch):
    """Blocks of 25 tasks over a 5-column pool give the single launch's outputs exactly and the numba scan's to 1e-9."""
    U, tasks, yr, codes, ky, clips = sweeps._scan_inputs(8_000, 5)
    whole = scan_offset_products_gpu(U, tasks, yr, codes, ky, sweeps.N_BINS, clips)
    assert whole is not None
    import mlframe.feature_selection.filters._device_sort_safe as dss

    monkeypatch.setattr(dss, "device_room", lambda cp: 1)  # forces the memory-bounded path
    monkeypatch.setattr(db, "first_block", lambda *a, **k: 25)
    blocked = scan_offset_products_gpu(U, tasks, yr, codes, ky, sweeps.N_BINS, clips)
    assert blocked is not None
    assert np.array_equal(whole[0], blocked[0], equal_nan=True) and np.array_equal(whole[1], blocked[1], equal_nan=True)
    out_shift = np.empty((len(tasks), 2))
    out_mi = np.empty((len(tasks), 2, 1 + N_BASELINES))
    scan_offset_products(U, tasks, yr, codes, ky, sweeps.N_BINS, clips, out_shift, out_mi)
    np.testing.assert_allclose(blocked[1], out_mi, rtol=0, atol=1e-9)
