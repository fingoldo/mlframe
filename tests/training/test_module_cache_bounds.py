"""Sensors for the module-level cache bounds added by the race+state-leak audit wave.

Covers:
- _PROXY_CLS_CACHE: OrderedDict + lock + cap 128
- _PLOT_IDX_CACHE: OrderedDict + lock + cap 256
- _INIT_SIG_CACHE: WeakKeyDictionary keyed on class (no id-recycle hazard)
"""

from __future__ import annotations

import threading

import numpy as np


# ---- _PROXY_CLS_CACHE ------------------------------------------------------


def test_proxy_cls_cache_bounded():
    """Wrap 200 distinct dynamic types; cache must stay at cap (128) not grow unbounded."""
    from mlframe.training.logging_transformers import (
        wrap_with_logging,
        _PROXY_CLS_CACHE,
        _PROXY_CLS_CACHE_MAX,
    )

    _PROXY_CLS_CACHE.clear()

    class _Dummy:
        """Groups tests covering dummy."""
        def fit(self, X, y=None):
            """Fit."""
            return self

        def transform(self, X):
            """Transform."""
            return X

    # Dynamically build 200 distinct types so cache_key (cls, label, methods) differs each call.
    for i in range(200):
        DynCls = type(f"_Dyn{i}", (_Dummy,), {})
        obj = DynCls()
        wrap_with_logging(obj)

    assert len(_PROXY_CLS_CACHE) <= _PROXY_CLS_CACHE_MAX, f"_PROXY_CLS_CACHE grew unbounded: {len(_PROXY_CLS_CACHE)} > cap {_PROXY_CLS_CACHE_MAX}"


def test_proxy_cls_cache_thread_safe():
    """Concurrent threads building proxies must not corrupt the cache."""
    from mlframe.training.logging_transformers import wrap_with_logging

    class _Dummy:
        """Groups tests covering dummy."""
        def fit(self, X, y=None):
            """Fit."""
            return self

        def transform(self, X):
            """Transform."""
            return X

    errors = []

    def _worker(_n: int):
        """Worker."""
        try:
            for _ in range(50):
                wrap_with_logging(_Dummy())
        except Exception as _exc:
            errors.append((_n, _exc))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent wraps raised: {errors}"


# ---- _PLOT_IDX_CACHE -------------------------------------------------------


def test_plot_idx_cache_bounded():
    """Plot idx cache bounded."""
    from mlframe.training.evaluation import (
        _get_cached_plot_idx,
        _PLOT_IDX_CACHE,
        _PLOT_IDX_CACHE_MAX,
    )

    _PLOT_IDX_CACHE.clear()

    # 300 distinct (n, sample_size, seed) tuples; cache stays at cap.
    for n in range(1000, 1300):
        _get_cached_plot_idx(n=n, sample_size=100, seed=42)

    assert len(_PLOT_IDX_CACHE) <= _PLOT_IDX_CACHE_MAX, f"_PLOT_IDX_CACHE grew unbounded: {len(_PLOT_IDX_CACHE)} > cap {_PLOT_IDX_CACHE_MAX}"


def test_plot_idx_cache_returns_same_idx_for_same_key():
    """LRU touch must not invalidate the cached value (still returns same array)."""
    from mlframe.training.evaluation import _get_cached_plot_idx

    a = _get_cached_plot_idx(n=500, sample_size=50, seed=7)
    b = _get_cached_plot_idx(n=500, sample_size=50, seed=7)
    np.testing.assert_array_equal(a, b)


def test_plot_idx_cache_thread_safe():
    """Plot idx cache thread safe."""
    from mlframe.training.evaluation import _get_cached_plot_idx

    errors = []
    results = {}
    lock = threading.Lock()

    def _worker(_n: int):
        """Worker."""
        try:
            for _ in range(30):
                idx = _get_cached_plot_idx(n=1000, sample_size=100, seed=_n)
                with lock:
                    results.setdefault(_n, []).append(idx)
        except Exception as _exc:
            errors.append((_n, _exc))

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent plot_idx raised: {errors}"
    # Same seed -> same idx across threads
    for idxs in results.values():
        for _i in idxs[1:]:
            np.testing.assert_array_equal(idxs[0], _i)


# ---- _INIT_SIG_CACHE -------------------------------------------------------


def test_init_sig_cache_weakkey_clears_on_class_gc():
    """When a class is GC'd, the cache entry must vanish too -- prevents id-recycle hazard."""
    import gc
    from mlframe.training.core._phase_train_one_target import (
        _cached_init_params,
        _INIT_SIG_CACHE,
    )

    class _ShimA:
        """Groups tests covering shim a."""
        def __init__(self, foo, bar):
            pass

    params_a = _cached_init_params(_ShimA)
    assert params_a == {"foo", "bar"}
    assert _ShimA in _INIT_SIG_CACHE  # cached

    n_before = len(_INIT_SIG_CACHE)
    del _ShimA
    gc.collect()
    n_after = len(_INIT_SIG_CACHE)
    assert n_after < n_before, f"WeakKeyDictionary should evict _ShimA after del+gc: before={n_before} after={n_after}"


def test_init_sig_cache_no_id_recycle_collision():
    """Two distinct classes that share an id() (after GC) must NOT cross-cache.

    This is the bug the WeakKeyDictionary fix prevents: id()-keyed caching could
    return the wrong signature if a recycled id collided with a newly-built shim class.
    """
    import gc
    from mlframe.training.core._phase_train_one_target import _cached_init_params

    class _ShimX:
        """Groups tests covering shim x."""
        def __init__(self, alpha):
            pass

    params_x = _cached_init_params(_ShimX)
    assert params_x == {"alpha"}
    del _ShimX
    gc.collect()

    # Now build a DIFFERENT class with potentially-recycled id but different signature.
    class _ShimY:
        """Groups tests covering shim y."""
        def __init__(self, beta, gamma):
            pass

    params_y = _cached_init_params(_ShimY)
    assert params_y == {"beta", "gamma"}, f"id-recycle bug regression: _ShimY returned _ShimX's cached signature. Got: {params_y}"
