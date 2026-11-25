"""500000 x 1000 fixes: the candidate-pool fingerprints are taken in blocks (no second copy of the whole pool) and the threaded device scorers take turns on a pool instead of oversubscribing it."""
import threading
import time

import numpy as np
import pytest


def _cupy():
    """cupy with a usable device, or skip."""
    cp = pytest.importorskip("cupy")
    try:  # audit-exempt: except_skip_masks_call_under_test -- optional CUDA device guard: the probe decides whether the module can run, it is not the code under test
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    return cp


def test_block_fingerprints_equal_one_stack(monkeypatch):
    """pool_fingerprints in blocks of a few columns gives exactly the fingerprints of one stack of the whole pool, for mixed integer dtypes."""
    cp = _cupy()
    from mlframe.feature_selection.filters import _bin_fingerprint_gpu as B

    rng = np.random.default_rng(4)
    n = 5000
    cols = [cp.asarray(rng.integers(0, 10, n).astype(dt)) for dt in (np.int8, np.int16, np.int32, np.int8, np.uint8, np.int64) * 5]
    whole = B._block_fingerprints(cp, cols)
    monkeypatch.setattr(B, "_STACK_BLOCK_BYTES", n * 3)  # about three columns per block
    blocked = B.pool_fingerprints(cp, cols)
    assert blocked == whole
    assert len(blocked) == len(cols)


def test_device_exclusive_serialises_threads_and_is_reentrant():
    """Two threads in a decorated routine never overlap, and a decorated routine may call another decorated one."""
    cp = _cupy()
    from mlframe.feature_selection.filters._device_sort_safe import device_exclusive

    state = {"inside": 0, "max": 0}
    guard = threading.Lock()

    @device_exclusive
    def inner():
        """Count how many threads are inside at once."""
        with guard:
            state["inside"] += 1
            state["max"] = max(state["max"], state["inside"])
        time.sleep(0.05)
        with guard:
            state["inside"] -= 1
        return 1

    @device_exclusive
    def outer():
        """Call another decorated routine (re-entrant)."""
        return inner() + inner()

    threads = [threading.Thread(target=outer) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["max"] == 1
    assert cp.cuda.runtime.getDeviceCount() >= 1
