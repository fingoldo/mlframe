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


def test_lightning_external_callbacks_cached_per_group(monkeypatch):
    """First call hits Lightning's slow path; second call must be near-instant
    (>=50x faster) and return an equivalent (but freshly-copied) list."""
    # Trigger the mlframe monkey-patch by importing the neural base.
    import mlframe.training.neural.base  # noqa: F401
    from lightning.fabric.utilities.registry import _load_external_callbacks

    group = "lightning.pytorch.callbacks_factory"
    # First call (might already be warm from earlier tests in the session;
    # call once to ensure cache populated, then measure subsequent calls).
    _load_external_callbacks(group)

    # The cache exists to stop Lightning re-scanning every installed distribution, and that scan goes
    # through ``importlib.metadata.entry_points``. Counting those is exact; a 50ms budget for 10 calls is
    # 5ms each, which the nightly coverage job's line tracing or a contended worker inflates on healthy
    # code, and which an uninstalled cache could still slip under on a machine with few distributions.
    import importlib.metadata

    scans = []
    real_entry_points = importlib.metadata.entry_points

    def _counting_entry_points(*args, **kwargs):
        """Record each full distribution scan."""
        scans.append(1)
        return real_entry_points(*args, **kwargs)

    monkeypatch.setattr(importlib.metadata, "entry_points", _counting_entry_points)
    for _ in range(10):
        _load_external_callbacks(group)
    assert not scans, f"10 cached calls triggered {len(scans)} entry-point scans; the cache wrapper is not installed"


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

    assert getattr(_lf_registry, "_mlframe_callback_cache_installed", False), "expected _mlframe_callback_cache_installed marker after import"


def test_lightning_cache_rebound_in_caller_modules():
    """The cache MUST be rebound in every Lightning module that imported
    ``_load_external_callbacks`` by name. Lightning's
    ``callback_connector.py`` and ``fabric.py`` both do
    ``from lightning.fabric.utilities.registry import _load_external_callbacks``
    at module import time, which binds the ORIGINAL function into the caller's
    namespace -- mutating ``_lf_registry._load_external_callbacks`` after that
    does NOT affect those local bindings. c0119 iter259 profile attributed
    3.73s / 12 calls to the uncached function because the iter189 patch only
    touched the registry, not the importers.

    This test pins the rebind: every Lightning module that holds a reference
    must point at the cached wrapper, not the original.
    """
    import sys
    import mlframe.training.neural.base  # noqa: F401  # installs the patch

    expected_name = "_load_external_callbacks_cached"
    mismatched = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("lightning"):
            continue
        f = getattr(mod, "_load_external_callbacks", None)
        if f is None:
            continue
        if getattr(f, "__name__", None) != expected_name:
            mismatched.append((mod_name, getattr(f, "__name__", "?")))
    assert not mismatched, (
        f"these Lightning modules still hold the uncached function: "
        f"{mismatched}. Lightning may have added a new import site -- extend "
        f"the rebind loop in mlframe/training/neural/base.py."
    )
