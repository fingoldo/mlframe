"""The two id-keyed fit caches hold their arrays weakly, evict safely under threads and are emptied on demand (tracker 2026-11-02, STATE5-2)."""

import gc
import threading
import weakref

import numpy as np

from mlframe.feature_selection.filters import _grouped_coerce_shared as gcs
from mlframe.feature_selection.filters._mrmr_fit_impl import _eng_dedup_batch_corr as edb


def test_sort_order_cache_does_not_keep_the_column_alive():
    """After the caller drops a column, the cache no longer holds it: a weak reference to it dies."""
    edb.clear_finite_sort_order_cache()
    col = np.random.default_rng(0).normal(size=1000)
    ref = weakref.ref(col)
    first = edb.finite_sort_order(col)
    assert edb.finite_sort_order(col) is first
    del col
    gc.collect()
    assert ref() is None


def test_a_dead_entry_never_answers_for_a_new_array_with_the_same_id():
    """A different array gets its own order even if it happens to take over a freed array's id."""
    edb.clear_finite_sort_order_cache()
    a = np.array([3.0, 1.0, 2.0])
    edb.finite_sort_order(a)
    del a
    gc.collect()
    b = np.array([9.0, 8.0, 7.0])
    assert edb.finite_sort_order(b).tolist() == [2, 1, 0]


def test_eviction_under_threads_does_not_raise():
    """Many threads filling a small cache at once never hit a KeyError from two of them evicting the same oldest key."""
    edb.clear_finite_sort_order_cache()
    errors: list[BaseException] = []
    arrays = [np.random.default_rng(i).normal(size=200) for i in range(300)]

    def work(chunk):
        """Sort-order every array of the chunk."""
        try:
            for arr in chunk:
                edb.finite_sort_order(arr)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(arrays[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(edb._ORDER_CACHE) <= edb._ORDER_CACHE_MAX


def test_clear_functions_empty_both_caches():
    """The end-of-fit clear leaves nothing behind."""
    keep = np.array(["a", "b", "a", None], dtype=object)
    gcs._prepared_keys(keep)
    edb.finite_sort_order(np.array([2.0, 1.0]))
    gcs.clear_key_prep_cache()
    edb.clear_finite_sort_order_cache()
    assert gcs._KEY_PREP == {} and edb._ORDER_CACHE == {}


def test_prepared_keys_cache_does_not_keep_the_key_array_alive():
    """The prepared-keys cache holds its key array weakly too."""
    gcs.clear_key_prep_cache()
    keys = np.array(["x", "y", "x"], dtype=object)
    ref = weakref.ref(keys)
    gcs._prepared_keys(keys)
    del keys
    gc.collect()
    assert ref() is None
