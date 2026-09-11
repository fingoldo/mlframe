"""No test reloads a module, or drops it from sys.modules, without a restore in the same scope.

`py_ci_shared.module_reload_safety`: a reload rebinds the module's names, so tests that imported it earlier
hold the old objects and later ones the new, and `isinstance` and class-level caches disagree far from the
cause. Each site needs a restore reachable from its own function or fixture, or a subprocess.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.module_reload_safety import assert_no_unpaired_reloads

TESTS_DIR = Path(__file__).resolve().parents[1]

# Files that reload only non-mlframe stub modules, where rebinding splits no mlframe class identity.
_KNOWN_STUB_ONLY_FILES = ("training/pipeline/test_pipeline_json_roundtrip_cache.py",)

# Modules owning a mutable singleton (cache, registry, lock): a __dict__ restore does not rebuild an object
# importers captured by reference, so an unpaired reload of one is reported as needing a subprocess.
_SINGLETON_OWNING_MODULES = (
    "mlframe.feature_selection.filters.mrmr",
    "mlframe.training.phases",
    "mlframe.training.composite.cache",
    "mlframe.training.suite_artefact_cache",
    "mlframe.system.kernel_tuning_cache",
)


def test_no_unpaired_module_reload_in_tests() -> None:
    """Every reload or sys.modules removal in tests has a restore in its own scope."""
    assert_no_unpaired_reloads(TESTS_DIR, stub_only_files=_KNOWN_STUB_ONLY_FILES, singleton_modules=_SINGLETON_OWNING_MODULES)
