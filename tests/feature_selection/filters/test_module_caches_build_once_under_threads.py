"""Threads asking a module-level cache for the same entry build it once and all get the same object (tracker 2026-11-02, unlocked-cache gate)."""

import threading

import numpy as np

from mlframe.feature_selection.filters import _orthogonal_bootstrap_mi_fe as bm


def test_seeded_bootstrap_rows_draws_one_list_for_concurrent_callers():
    """Eight threads asking for the same seeded resamples receive the identical list object, drawn once."""
    bm._REPS_CACHE.clear()
    y = np.random.default_rng(1).integers(0, 2, 400)
    results: list = []
    barrier = threading.Barrier(8)

    def work():
        """Ask for the resamples at the same moment as the other threads."""
        barrier.wait()
        results.append(bm.seeded_bootstrap_rows(7, y, 400, 400, 4))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    assert all(r is results[0] for r in results)


def test_release_during_use_does_not_raise():
    """Releasing the cache while another thread fills it never raises and always reports a non-negative size."""
    bm._REPS_CACHE.clear()
    y = np.random.default_rng(2).integers(0, 2, 300)
    errors: list = []

    def fill():
        """Fill the cache repeatedly with different seeds."""
        try:
            for seed in range(30):
                bm.seeded_bootstrap_rows(seed, y, 300, 300, 2)
        except BaseException as exc:
            errors.append(exc)

    def release():
        """Release the cache repeatedly."""
        try:
            for _ in range(200):
                assert bm._release_reps_cache() >= 0
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=fill), threading.Thread(target=release)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
