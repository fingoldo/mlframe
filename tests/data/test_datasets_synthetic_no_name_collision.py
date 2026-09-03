"""Meta-test: the two star-imported submodules of `mlframe.data` must not export a common name.

`mlframe/data/__init__.py` runs `from ... import *` over `datasets` and then `synthetic`, so on any
name collision the second import silently wins and `mlframe.data.<name>` resolves to the wrong object.
`pyutilz.dev.code_audit`'s `uncurated_star_export` check skips absolute imports, so nothing else catches it.
"""

from __future__ import annotations

from types import ModuleType

import mlframe.data as data
import mlframe.data.datasets as datasets
import mlframe.data.synthetic as synthetic


def _star_surface(module) -> set:
    """Return the names a `from module import *` would bind: its `__all__` if declared, else its public globals."""

    declared = getattr(module, "__all__", None)
    if declared is not None:
        return set(declared)
    return {name for name in vars(module) if not name.startswith("_")}


def test_datasets_and_synthetic_star_surfaces_do_not_collide():
    """A shared name would be silently shadowed by the second star import in `mlframe.data.__init__`."""

    collisions = _star_surface(datasets) & _star_surface(synthetic)
    assert collisions == set(), f"mlframe.data star-import collision, second import wins: {sorted(collisions)}"


def test_datasets_declares_a_literal_all():
    """vulture only honours a literal list/tuple of string constants; anything else flags the re-exports as unused."""

    assert isinstance(datasets.__all__, list)
    assert all(isinstance(name, str) for name in datasets.__all__)
    assert set(datasets.__all__) == {"get_sapp_dataset", "indicator", "showcase_pycaret_datasets"}


def test_data_package_exports_no_module_objects():
    """The public surface must be API only.

    Deriving `__all__` from `globals()` republishes whatever the star-imported submodules happened to import
    (`np`, `pd`, `stats`, `njit`, ...) plus this package's own `annotations`, which is how the collision above
    became possible in the first place: the shadowed names were all third-party module objects.
    """

    exported_modules = sorted(name for name in data.__all__ if isinstance(getattr(data, name), ModuleType))
    assert exported_modules == [], f"mlframe.data re-exports module objects, not API: {exported_modules}"


def test_synthetic_declares_a_literal_all():
    """Without an explicit `__all__` here, every public global leaks into `mlframe.data` and shadows `datasets`."""

    assert isinstance(synthetic.__all__, list)
    assert all(isinstance(name, str) for name in synthetic.__all__)
    assert set(synthetic.__all__) == {"assign_classes_from_probability", "generate_modelling_data", "sample_random_variable"}
