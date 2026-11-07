"""The device draw of the family null's class bit sets equals the host draw bit for bit, and the generator ends where a host draw leaves it."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _family_maxt_gate as G
from mlframe.feature_selection.filters._family_maxt_null_v2_njit import label_bitsets_from_index
from mlframe.feature_selection.filters._perm_labels_device import device_label_bits


def _device_ok() -> bool:
    """Whether a CUDA device answers."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # no driver: skipped
        return False


@pytest.mark.skipif(not _device_ok(), reason="no CUDA device")
@pytest.mark.parametrize("n_y", [2, 3])
def test_device_bits_equal_host_bits(n_y):
    """The same seed gives the same permutations and the same packed bits, for a row count that is not a multiple of 64."""
    rng_y = np.random.default_rng(3)
    n, m = 1000 + 37, 9
    y = rng_y.integers(0, n_y, n).astype(np.int64)
    words = (n + 63) // 64
    host_rng = np.random.default_rng(11)
    want = label_bitsets_from_index(y, G._permutation_block(host_rng, y, m), n_y, words)
    dev_rng = np.random.default_rng(11)
    base = int(dev_rng.integers(0, 2**31 - 1))
    got = device_label_bits(y, m, base, n_y, words)
    assert got is not None and np.array_equal(got, want)
    assert dev_rng.bit_generator.state == host_rng.bit_generator.state


@pytest.mark.skipif(not _device_ok(), reason="no CUDA device")
def test_block_helper_uses_the_device_above_the_row_floor_and_keeps_the_stream(monkeypatch):
    """Above the row floor the block comes from the device and equals the host block, and the generator state is the one a host draw leaves."""
    monkeypatch.setattr(G, "_DEVICE_LABELS_MIN_ROWS", 100)
    monkeypatch.setenv("MLFRAME_PERM_LABELS_DEVICE", "1")
    rng_y = np.random.default_rng(4)
    y = rng_y.integers(0, 2, 700).astype(np.int64)
    words = (700 + 63) // 64
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    G._label_bits_cache.clear()
    on_device = G._label_bits_block(a, y, 7, 2, words, 1, on_device=True)
    G._label_bits_cache.clear()
    on_host = G._label_bits_block(b, y, 7, 2, words, 2, on_device=False)
    assert np.array_equal(on_device, on_host)
    assert a.bit_generator.state == b.bit_generator.state
