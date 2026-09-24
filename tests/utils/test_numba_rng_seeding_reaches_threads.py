"""Seeding must reach numba's per-thread RNG, not just the calling thread's."""

import threading

import numpy as np
import pytest

pytest.importorskip("numba")
from numba import njit, prange

from mlframe.utils.misc import seed_numba_in_this_thread, seed_numba_worker_threads


@njit(parallel=True, cache=True)
def _parallel_draws(n):
    """One draw per iteration, spread over numba's worker threads."""
    out = np.empty(n)
    for i in prange(n):
        out[i] = np.random.random()
    return out


@njit(cache=True)
def _one_draw():
    """A single draw from this thread's numba RNG."""
    return np.random.random()


def test_a_prange_kernel_is_reproducible_after_seeding():
    """Seeding only the calling thread left every prange worker on its own entropy-seeded stream."""
    seed_numba_worker_threads(7)
    first = _parallel_draws(64)
    seed_numba_worker_threads(7)
    second = _parallel_draws(64)
    np.testing.assert_array_equal(first, second)


def test_a_different_seed_gives_a_different_stream():
    """The negative control: seeding is only meaningful if a different seed changes the draws."""
    seed_numba_worker_threads(7)
    first = _parallel_draws(64)
    seed_numba_worker_threads(8)
    assert not np.array_equal(first, _parallel_draws(64))


def test_a_threading_worker_can_seed_its_own_numba_rng():
    """A joblib backend="threading" worker is an ordinary thread with its own state; nothing else seeds it."""

    def draw(seed, out, i):
        """Seed this thread's numba RNG and record one draw."""
        seed_numba_in_this_thread(seed)
        out[i] = _one_draw()

    out = [None, None]
    threads = [threading.Thread(target=draw, args=(11, out, i)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert out[0] is not None and out[0] == out[1], "two threads seeded identically must draw identically"


def test_set_random_seed_covers_the_worker_threads():
    """The package-level seeder must reach numba's parallel worker threads, not just the calling thread."""
    from mlframe.utils.misc import set_random_seed

    set_random_seed(123)
    first = _parallel_draws(64)
    set_random_seed(123)
    np.testing.assert_array_equal(first, _parallel_draws(64))
