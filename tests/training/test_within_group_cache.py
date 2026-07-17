"""Regression: ``_numba_within_group_descending_rank`` must be
``@njit(cache=True)`` so the JIT compile cost is paid ONCE per
machine, not on every fresh process.

Pre-fix path (fuzz c0114_12268ceb LTR + 3 boosters):
1. ``train_mlframe_ranker_suite`` -> ``compute_dummy_baselines``
   (LTR target type) -> ``_compute_ltr_baselines`` -> two calls to
   ``_within_group_descending_index`` (one for val, one for test).
2. First call triggers numba JIT compile because
   ``@njit(cache=False)``. cProfile attribution: ``3.9 s`` cumtime
   in ``numba.core.dispatcher._compile_for_args``.
3. Every fresh process repeats the cost. The kernel itself is
   trivial (single argsort + sequential scan) so the compile cost
   ratio is ~100x the actual work cost.

Post-fix: ``@njit(cache=True)`` writes the compiled binary to
``__pycache__/`` on the first compile; subsequent processes
deserialise from disk (~60ms) instead of recompiling (~3-6s).

Empirical bench on this machine (n=8 integer gids):
- 1st call in process that DID the compile: ~6 s (compile cost).
- 1st call in fresh process with the disk cache present: ~60 ms.
- Speedup: ~100x on cold-start.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_numba_within_group_decorator_has_cache_true() -> None:
    """The kernel must be `@njit(cache=True)`. We can't inspect the
    decorator string at runtime, but numba dispatchers expose
    ``_cache`` (NullCache vs IndexDataCacheFile) which surfaces the
    flag's value."""
    pytest.importorskip("numba")
    from mlframe.training.baselines.dummy import _numba_within_group_descending_rank

    # Numba's CPUDispatcher stores the cache backend on ``_cache``.
    # The NullCache class signals cache=False; FunctionCache (or its
    # IndexDataCacheFile child) signals cache=True. Check by name.
    cache_obj = getattr(_numba_within_group_descending_rank, "_cache", None)
    assert cache_obj is not None, "numba dispatcher missing _cache attribute"
    cache_cls_name = type(cache_obj).__name__
    assert cache_cls_name != "NullCache", (
        f"_numba_within_group_descending_rank still has cache=False "
        f"(_cache class is {cache_cls_name}). Flip to @njit(cache=True) "
        "so the JIT compile cost is paid once per machine."
    )


def test_within_group_descending_index_behaviour_unchanged() -> None:
    """The cache=True flip is a perf change; the per-row output must
    be bit-identical to the cache=False baseline."""
    from mlframe.training.baselines.dummy import _within_group_descending_index

    # 3 groups, 2 rows each, then 2 more from groups 0 and 1.
    gids = np.array([0, 0, 1, 1, 2, 2, 0, 1], dtype=np.int64)
    out = _within_group_descending_index(gids, len(gids))
    # Expected per-group descending index: -within_group_position.
    # Group 0 rows are at indices [0, 1, 6] -> codes [-0, -1, -2].
    # Group 1 rows at [2, 3, 7] -> codes [-0, -1, -2].
    # Group 2 rows at [4, 5]    -> codes [-0, -1].
    expected = np.array([-0.0, -1.0, -0.0, -1.0, -0.0, -1.0, -2.0, -2.0])
    np.testing.assert_array_equal(out, expected)


def test_within_group_handles_string_keys_via_python_fallback() -> None:
    """String group_ids must still work via the Python-loop fallback
    (the kernel only handles integer dtypes; the wrapper falls back
    on `dtype.kind` not in {'i','u'})."""
    from mlframe.training.baselines.dummy import _within_group_descending_index

    gids = np.array(["a", "a", "b", "a", "b"], dtype=object)
    out = _within_group_descending_index(gids, len(gids))
    # "a" appears at positions 0,1,3 -> codes [-0,-1,-2].
    # "b" appears at positions 2,4   -> codes [-0,-1].
    expected = np.array([-0.0, -1.0, -0.0, -2.0, -1.0])
    np.testing.assert_array_equal(out, expected)


def test_within_group_empty_input_returns_empty_array() -> None:
    """n=0 short-circuit must not invoke the kernel (would compile
    on an empty array which the dispatcher handles fine, but the
    short-circuit saves a dispatch)."""
    from mlframe.training.baselines.dummy import _within_group_descending_index

    out = _within_group_descending_index(np.array([], dtype=np.int64), 0)
    assert out.shape == (0,)
