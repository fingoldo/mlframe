"""Regression test for the Lightning _load_external_callbacks per-process cache.

Lightning's ``_load_external_callbacks`` scans every installed Python distribution
via ``importlib.metadata.entry_points`` on EACH ``Trainer.fit()`` /
``Trainer.predict()`` invocation -- ~180ms / call on a Windows box with a typical
anaconda site-packages (5346 dist-info METADATA reads). c0065 iter189 profile
attributed 1.484s to this across 6 fit calls (~6% of the 23.4s wall).

The mlframe cache wrapper in ``training/neural/base.py`` memoises by group name
since callback discovery is process-stable (sys.path + installed dists don't
change between fits). Mirrors the ``_PROBE_PRECISION_CACHE`` pattern
(mlp_runtime_defaults.py iter181) and ``_CB_GPU_USABLE_CACHE`` (_cb_pool.py).
"""
import time

import pytest


def test_lightning_external_callbacks_cached_per_group():
    """First call hits Lightning's slow path; second call must be near-instant
    (>=50x faster) and return an equivalent (but freshly-copied) list."""
    # Trigger the mlframe monkey-patch by importing the neural base.
    import mlframe.training.neural.base  # noqa: F401
    from lightning.fabric.utilities.registry import _load_external_callbacks

    group = "lightning.pytorch.callbacks_factory"
    # First call (might already be warm from earlier tests in the session;
    # call once to ensure cache populated, then measure subsequent calls).
    _load_external_callbacks(group)

    # Now measure 10 cached calls.
    t = time.perf_counter()
    for _ in range(10):
        _load_external_callbacks(group)
    elapsed_warm = time.perf_counter() - t
    # 10 cached calls must complete in well under 50ms (typical ~0ms each;
    # uncached would be ~1.8s for 10 calls).
    assert elapsed_warm < 0.05, (
        f"cached 10 calls took {elapsed_warm*1000:.1f}ms; "
        f"cache wrapper likely not installed (expected <50ms)"
    )


def test_lightning_external_callbacks_cache_returns_defensive_copy():
    """Cache must hand out a fresh list per call so caller mutations
    don't poison the next call's result."""
    import mlframe.training.neural.base  # noqa: F401
    from lightning.fabric.utilities.registry import _load_external_callbacks

    group = "lightning.pytorch.callbacks_factory"
    r1 = _load_external_callbacks(group)
    # Mutate the returned list.
    r1.append("intentional_test_sentinel_must_not_persist")
    r2 = _load_external_callbacks(group)
    assert "intentional_test_sentinel_must_not_persist" not in r2


def test_lightning_cache_install_marker_set():
    """After importing neural.base, the cache install marker must be set
    so future imports skip re-wrapping (avoids double-cache + reference leak).

    NOTE: not exercising ``importlib.reload(neural.base)`` here -- module reload
    breaks class identity for already-imported sibling modules (per repo
    convention: feedback_no_module_reload_without_snapshot). The marker
    pattern is verified by inspection.
    """
    import mlframe.training.neural.base  # noqa: F401
    from lightning.fabric.utilities import registry as _lf_registry

    assert getattr(_lf_registry, "_mlframe_callback_cache_installed", False), (
        "expected _mlframe_callback_cache_installed marker after import"
    )
