"""Regression test for audit2 M1: the process-wide per-column discretization code cache is mutated under
joblib backend="threading" with no lock -> concurrent OrderedDict mutation could raise RuntimeError and the
module-global byte counter could drift. A threading.Lock now guards every mutation; assert that hammering the
cached wrapper from many threads (a) never raises, (b) stays bit-identical to the uncached kernel, and
(c) leaves the byte counter EXACTLY equal to the sum of the retained entries' bytes (no drift).
"""

import threading

import numpy as np

from mlframe.feature_selection.filters.discretization import discretize_2d_array
from mlframe.feature_selection.filters import _fe_subsample  # noqa: F401  (warms numba once)
from mlframe.feature_selection.filters.discretization import _discretization_dataset as dd


def _cached(arr):
    """Helper that cached."""
    return dd._discretize_2d_array_col_cached(
        arr,
        n_bins=8,
        method="quantile",
        min_ncats=2,
        dtype=np.int32,
        discretize_2d_array=discretize_2d_array,
    )


def test_concurrent_col_cache_is_correct_and_counter_does_not_drift():
    """Concurrent col cache is correct and counter does not drift."""
    rng = np.random.default_rng(0)
    # Overlapping column sets across threads so hits, misses and evictions all interleave.
    base = rng.standard_normal((2000, 12)).astype(np.float64)
    reference = discretize_2d_array(arr=base, n_bins=8, method="quantile", min_ncats=2, min_values=None, max_values=None, dtype=np.int32)

    dd.clear_numeric_code_cache()

    errors: list = []
    results: list = []

    def worker(cols):
        """Helper that worker."""
        try:
            for _ in range(4):
                sub = np.ascontiguousarray(base[:, cols])
                out = _cached(sub)
                results.append((cols, out))
        except Exception as e:  # pragma: no cover - only on a real race
            errors.append(e)

    col_sets = [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [0, 6, 7, 8], [8, 9, 10, 11], [1, 5, 9, 11]]
    threads = [threading.Thread(target=worker, args=(cs,)) for cs in col_sets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent cache access raised: {errors}"

    # Every thread's output must equal the reference kernel column-for-column (bit-identical by construction).
    for cols, out in results:
        for _i, c in enumerate(cols):
            assert np.array_equal(out[:, _i], reference[:, c]), f"cached codes for column {c} diverged from the kernel"

    # The byte counter must equal the actual retained bytes -- proves no double-count / lost-decrement drift.
    actual_bytes = sum(v.shape[0] * v.dtype.itemsize for v in dd._NUMERIC_CODE_CACHE.values())
    assert dd._NUMERIC_CODE_CACHE_BYTES == actual_bytes, (
        f"byte counter {dd._NUMERIC_CODE_CACHE_BYTES} != retained {actual_bytes} (accounting drifted under concurrency)"
    )
