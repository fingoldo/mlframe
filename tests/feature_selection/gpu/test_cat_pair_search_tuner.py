"""The categorical pair-search tuner entry: both backends agree, the gate reads it, and the device path keeps its input checks."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _cat_pair_search_tuner as tuner
from mlframe.feature_selection.filters._gpu_pairs import mi_direct_gpu_batched_pairs
from tests.feature_selection.gpu._gpu_support import has_device

pytestmark = pytest.mark.gpu


def test_host_and_device_pair_search_agree():
    """The njit prange kernel and the batched device kernels give the same joint MI for every pair, and the MI is not degenerate."""
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    args = tuner._sweep_inputs({"n_samples": 20_000})
    host, device = tuner._sweep_host(*args), tuner._sweep_device(*args)
    assert host.shape == device.shape == (1770,)
    assert float(host.max()) > 0.01
    assert np.abs(host - device).max() < 1e-12


def test_row_mark_decides_without_a_tuner_entry(monkeypatch):
    """With no tuner entry the answer is the row mark."""
    monkeypatch.setattr(tuner, "_SPEC", None)
    monkeypatch.setattr(tuner, "DEVICE_FALLBACK_ROWS", 1000)
    assert tuner.cat_pair_search_on_device(1000) is True
    assert tuner.cat_pair_search_on_device(999) is False


@pytest.mark.parametrize("bad, message", [(5, "contains 5"), (-1, "negative")])
def test_out_of_range_codes_are_rejected(bad, message):
    """A code past the declared bin count, or a negative one, raises before any launch (the range check now runs on the transposed device copy)."""
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, size=(500, 3)).astype(np.int32)
    codes[7, 1] = bad
    y = rng.integers(0, 2, 500).astype(np.int32)
    with pytest.raises(ValueError, match=message):
        mi_direct_gpu_batched_pairs(codes, np.array([0, 1]), np.array([1, 2]), np.full(3, 4, dtype=np.int32), y, np.array([0.5, 0.5]), dtype=np.int32)
