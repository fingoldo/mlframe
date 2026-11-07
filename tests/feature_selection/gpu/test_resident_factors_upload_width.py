"""The resident code matrix is uploaded at its own width and widened to int32 on the device: same device values for every input width, and no 4-byte host copy for narrow input."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters.info_theory import _cmi_resident_cache as R


def _device_ok() -> bool:
    """Whether a CUDA device answers."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no driver: skipped
        return False


pytestmark = pytest.mark.skipif(not _device_ok(), reason="no CUDA device")


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8])
def test_device_matrix_is_int32_with_the_same_values(dtype):
    """Every input width lands on the device as the same int32 matrix."""
    rng = np.random.default_rng(0)
    nbins = np.array([5, 9, 3, 12])
    host = np.stack([rng.integers(0, int(b), 400) for b in nbins], axis=1).astype(dtype)
    dev = R._widen_on_device(cp, R._host_codes_for_upload(host))
    assert dev.dtype == cp.int32
    np.testing.assert_array_equal(cp.asnumpy(dev), host.astype(np.int32))


def test_narrow_input_is_not_copied_to_int32_on_the_host():
    """An int8 matrix goes up as int8 (the host helper returns the matrix itself), a wider unsigned one is cast."""
    host8 = np.zeros((10, 3), dtype=np.int8)
    assert R._host_codes_for_upload(host8) is host8 or np.shares_memory(R._host_codes_for_upload(host8), host8)
    assert R._host_codes_for_upload(host8).dtype == np.int8
    assert R._host_codes_for_upload(np.zeros((10, 3), dtype=np.uint16)).dtype == np.int32
    assert R._host_codes_for_upload(np.zeros((10, 3), dtype=np.int64)).dtype == np.int32


def test_out_of_range_code_is_still_refused():
    """The one-time guard works on the narrow host matrix: a code at or above its column's bin count raises."""
    nbins = np.array([4, 4])
    host = np.array([[1, 2], [3, 4]], dtype=np.int8)
    with pytest.raises(ValueError):
        R._assert_codes_in_range_2d_per_column(R._host_codes_for_upload(host), nbins)
