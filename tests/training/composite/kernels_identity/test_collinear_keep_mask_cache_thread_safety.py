"""TRAINING_COMPOSITE_DISCOVERY-2 regression test: _KEEP_MASK_CACHE's read path (get + move_to_end) must
be protected by the same lock as its write/evict path, so concurrent near_collinear_keep_mask_fast calls
from multiple threads can't corrupt the shared module-level LRU cache.

The bug (fixed): the read path called .get() + move_to_end() unlocked, while the write/evict path
(popitem + __setitem__) was already lock-protected -- OrderedDict.move_to_end() mutates the dict's
internal doubly-linked list, so an unlocked reader racing a writer's popitem()/__setitem__() risks a
KeyError or a corrupted LRU order.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from mlframe.training.composite.discovery import _collinear_numba as _cn
from mlframe.training.composite.discovery._collinear_numba import near_collinear_keep_mask_fast
from mlframe.training.composite.discovery._eval_stats import _near_collinear_keep_mask_numpy

pytestmark = pytest.mark.fast


def _call(fm, thr):
    """Invoke near_collinear_keep_mask_fast with the numpy reference fallback wired in."""
    return near_collinear_keep_mask_fast(fm, corr_threshold=thr, reference_fn=_near_collinear_keep_mask_numpy)


def test_concurrent_cache_hits_and_evictions_do_not_raise_or_corrupt(monkeypatch):
    """Many threads repeatedly hitting/evicting a small-capacity cache must never raise (KeyError from an
    unlocked read racing an evict) and every returned mask must still validate against the numpy reference."""
    # Small cache so eviction pressure is high and the race window matters within a short test.
    monkeypatch.setattr(_cn, "_KEEP_MASK_CACHE_MAX_ENTRIES", 4)
    _cn._KEEP_MASK_CACHE.clear()

    rng = np.random.default_rng(0)
    matrices = [np.ascontiguousarray(rng.normal(size=(50, 6)), dtype=np.float64) for _ in range(10)]
    thr = 0.9

    errors: list[BaseException] = []
    results: list[tuple[int, np.ndarray]] = []
    lock = threading.Lock()

    def _worker(idx):
        """Repeatedly call near_collinear_keep_mask_fast across the shared matrix pool, forcing cache
        churn (hits + evictions) across threads, recording any exception and the (matrix_index, mask) pair."""
        try:
            for i in range(30):
                mat_idx = (idx + i) % len(matrices)
                out = _call(matrices[mat_idx], thr)
                with lock:
                    results.append((mat_idx, out))
        except BaseException as e:  # must capture ANY exception a race could raise
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"concurrent cache access raised: {errors}"
    assert results, "expected at least one successful result"
    references = [_near_collinear_keep_mask_numpy(fm, corr_threshold=thr) for fm in matrices]
    for mat_idx, out in results:
        np.testing.assert_array_equal(out, references[mat_idx])
