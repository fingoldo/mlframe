"""FE_ROOT_A-7: _recursion_autotune's module-level kernel_tuner registration loop claimed (in an adjacent
comment) to be 'wrapped so a missing pyutilz / circular import never breaks the dispatcher', but had no
actual try/except -- a missing/broken pyutilz.performance.kernel_tuning.registry would abort the whole
module import instead of degrading gracefully."""

from __future__ import annotations

import importlib

import mlframe.feature_engineering._recursion_autotune as _recursion_autotune_module


def test_module_reload_survives_missing_kernel_tuner(monkeypatch):
    """Reloading _recursion_autotune with kernel_tuner made unimportable must not raise -- the
    module-level registration block must degrade gracefully instead of aborting the import."""
    import pyutilz.performance.kernel_tuning.registry as registry_module

    def _raise(*args, **kwargs):
        """Stand-in for kernel_tuner that always raises ImportError."""
        raise ImportError("synthetic: kernel_tuner unavailable")

    monkeypatch.setattr(registry_module, "kernel_tuner", _raise)
    try:
        reloaded = importlib.reload(_recursion_autotune_module)
        assert hasattr(reloaded, "ensure_recursion_tuning")
    finally:
        # Restore the real kernel_tuner and re-reload so later tests in the same process see the
        # normal, correctly-registered module state (module-level singletons must not leak).
        monkeypatch.undo()
        importlib.reload(_recursion_autotune_module)
