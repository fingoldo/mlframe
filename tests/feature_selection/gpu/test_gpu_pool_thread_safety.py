"""The shared GPU buffer pool must not be used by two threads at once.

`_GPU_POOL` is a module singleton of device buffers, and the FE pair sweep dispatches its workers with joblib
`backend="threading"`. Concurrent `mi_direct_gpu` calls interleaved on those buffers: one thread filled `classes_x`
while another reallocated `joint_counts` for a different `nbins_y` or overwrote the same arrays, so the first thread's
joint histogram was built from the second thread's data and the MI came back silently wrong - a junk engineered pair
admitted or a good one dropped, differently on every run.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import gpu as fs_gpu


def _factors(seed: int, n: int = 3000, nbins: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, nbins, size=(n, 2)).astype(np.int32)


def test_the_pool_is_held_for_the_whole_block_not_just_the_allocation(monkeypatch):
    """A probe inside `ensure()` must find the lock already taken by its own thread and unavailable to another."""
    held_elsewhere: list[bool] = []
    original_ensure = fs_gpu._GPU_POOL.ensure

    def probing_ensure(*args, **kwargs):
        done = threading.Event()

        def other_thread():
            # False means another thread could NOT take the lock, i.e. this block is exclusive.
            acquired = fs_gpu._GPU_POOL_LOCK.acquire(blocking=False)
            if acquired:
                fs_gpu._GPU_POOL_LOCK.release()
            held_elsewhere.append(not acquired)
            done.set()

        t = threading.Thread(target=other_thread)
        t.start()
        done.wait(timeout=10)
        t.join(timeout=10)
        return original_ensure(*args, **kwargs)

    monkeypatch.setattr(fs_gpu._GPU_POOL, "ensure", probing_ensure)
    fs_gpu.mi_direct_gpu(_factors(0), (0,), (1,), (4, 4), npermutations=4, base_seed=1)
    assert held_elsewhere == [True], "the pool block must be exclusive while a thread is inside it"


def test_concurrent_callers_get_the_same_answers_as_serial_ones():
    """Different bin counts per caller is the shape that reallocates `joint_counts` mid-flight."""
    beds = [(_factors(s, nbins=nb), nb) for s, nb in ((1, 4), (2, 6), (3, 5), (4, 8))]
    expected = [fs_gpu.mi_direct_gpu(f, (0,), (1,), (nb, nb), npermutations=8, base_seed=11)[0] for f, nb in beds]

    got: list[float | None] = [None] * len(beds)
    barrier = threading.Barrier(len(beds))

    def work(i):
        f, nb = beds[i]
        barrier.wait(timeout=30)
        got[i] = fs_gpu.mi_direct_gpu(f, (0,), (1,), (nb, nb), npermutations=8, base_seed=11)[0]

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(beds))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "a worker did not finish: the pool lock must not deadlock"
    for i, (exp, actual) in enumerate(zip(expected, got)):
        assert actual == pytest.approx(exp, rel=1e-9), f"bed {i}: concurrent MI {actual} != serial MI {exp}"
