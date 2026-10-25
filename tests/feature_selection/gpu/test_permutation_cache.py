"""The permutation stream is cached by content, with bounded memory, and a cached batch is the same array the generator returned."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gpu_host_permutations as hp


@pytest.fixture(autouse=True)
def _clean_caches():
    """Every test starts and ends with empty caches."""
    hp.clear_permutation_caches()
    yield
    hp.clear_permutation_caches()


def _target(n=500, k=4, seed=0):
    """Integer class codes of a target."""
    return np.random.default_rng(seed).integers(0, k, size=n).astype(np.int32)


def test_repeated_request_returns_the_same_array_and_equals_a_fresh_draw():
    """Same content, seed and slice: the second call is the cached array, and it equals what the generator draws without the cache."""
    y = _target()
    first = hp.host_permuted_y_batch(y, 7, 0, 16)
    again = hp.host_permuted_y_batch(y.copy(), 7, 0, 16)  # equal content in a different buffer
    assert again is first
    hp.clear_permutation_caches()
    fresh = hp.host_permuted_y_batch(y, 7, 0, 16)
    assert fresh is not first
    np.testing.assert_array_equal(fresh, first)


def test_key_covers_content_seed_slice_and_count():
    """Any change of the target, seed, first permutation or count gives a different batch, never a stale hit."""
    y = _target()
    base = hp.host_permuted_y_batch(y, 7, 0, 8)
    y2 = y.copy()
    y2[0] = (y2[0] + 1) % 4
    assert hp.host_permuted_y_batch(y2, 7, 0, 8) is not base
    assert hp.host_permuted_y_batch(y, 8, 0, 8) is not base
    assert hp.host_permuted_y_batch(y, 7, 8, 8) is not base
    assert hp.host_permuted_y_batch(y, 7, 0, 9) is not base
    assert hp.host_permuted_y_batch(y, None, 0, 8) is hp.host_permuted_y_batch(y, 0, 0, 8)


def test_returned_batch_is_read_only():
    """A shared batch cannot be modified by a caller."""
    out = hp.host_permuted_y_batch(_target(), 1, 0, 4)
    assert not out.flags.writeable
    with pytest.raises(ValueError):
        out[0, 0] = 99


def test_host_cache_stays_within_its_budget(monkeypatch):
    """Old batches are dropped once the total exceeds the budget; a batch larger than the budget is not cached."""
    y = _target(n=1000)
    one = hp.host_permuted_y_batch(y, 0, 0, 10).nbytes
    hp.clear_permutation_caches()
    monkeypatch.setattr(hp, "HOST_PERM_CACHE_BYTES", int(2.5 * one))
    for first in range(0, 100, 10):
        hp.host_permuted_y_batch(y, 0, first, 10)
    assert hp._HOST_CACHE_BYTES <= hp.HOST_PERM_CACHE_BYTES
    assert len(hp._HOST_CACHE) == 2
    monkeypatch.setattr(hp, "HOST_PERM_CACHE_BYTES", one - 1)
    hp.clear_permutation_caches()
    hp.host_permuted_y_batch(y, 0, 0, 10)
    assert len(hp._HOST_CACHE) == 0


def test_device_copy_is_reused_and_matches_the_host_batch():
    """The device copy of a cached host batch is uploaded once; a writable array is not cached."""
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("no cupy device")
    except Exception:  # nosec B110 - no usable device
        pytest.skip("no cupy device")
    host = hp.host_permuted_y_batch(_target(), 3, 0, 8)
    first = hp.device_array_cached(cp, host)
    assert hp.device_array_cached(cp, host) is first
    np.testing.assert_array_equal(cp.asnumpy(first), host)
    writable = np.array(host)
    assert hp.device_array_cached(cp, writable) is not hp.device_array_cached(cp, writable)


def test_device_cache_stays_within_its_budget(monkeypatch):
    """Device copies are evicted oldest first once the budget is exceeded."""
    cp = pytest.importorskip("cupy")
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("no cupy device")
    except Exception:  # nosec B110 - no usable device
        pytest.skip("no cupy device")
    y = _target(n=1000)
    one = hp.host_permuted_y_batch(y, 0, 0, 10).nbytes
    monkeypatch.setattr(hp, "DEVICE_PERM_CACHE_BYTES", int(2.5 * one))
    for first in range(0, 60, 10):
        hp.device_array_cached(cp, hp.host_permuted_y_batch(y, 0, first, 10))
    assert hp._DEVICE_CACHE_BYTES <= hp.DEVICE_PERM_CACHE_BYTES
    assert len(hp._DEVICE_CACHE) == 2
