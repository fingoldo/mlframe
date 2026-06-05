"""Regression unit test for /loop iter 5 optimization.

Hotspot: ``_setup_helpers.py`` previously had ``from mlframe.feature_selection.filters
import MRMR`` at module top, which transitively loads numba kernels + sklearn
estimators + filter wrappers. Isolated cost: ~16s on first import (measured
3-trial median in a fresh process with mlframe.training already loaded so the
MRMR subgraph is the only new work).

Fix: deferred the MRMR import into the ``if use_mrmr_fs:`` branch inside
``_build_pre_pipelines`` -- mirrors the pre-existing BorutaShap pattern in
the same function. Default ``use_mrmr_fs=False`` means most users avoid the
~16s tax on first call entirely.

This test asserts:
1. Module-level ``MRMR`` symbol is no longer re-exported from
   ``_setup_helpers`` -- the only top-level reference is the TYPE_CHECKING
   guard which is invisible at runtime.
2. After importing ``_setup_helpers`` fresh, ``mlframe.feature_selection.filters``
   is NOT yet in ``sys.modules`` (proves the deferral is real).
3. Calling ``_build_pre_pipelines(use_mrmr_fs=False, ...)`` does NOT trigger
   the import either (user opt-out path stays cheap).
4. Calling ``_build_pre_pipelines(use_mrmr_fs=True, ...)`` DOES trigger the
   import (user opt-in path still works correctly).
"""
from __future__ import annotations

import importlib
import sys

import pytest


_FILTERS_MODULE = "mlframe.feature_selection.filters"


def _force_clean_mrmr_modules():
    """Pop mlframe.feature_selection.filters and its submodules from sys.modules
    so the next import counts as fresh. Avoids ``importlib.reload`` to keep the
    test cheap (~ms) and not re-trigger numba JIT warmup.
    """
    for name in list(sys.modules):
        if name == _FILTERS_MODULE or name.startswith(_FILTERS_MODULE + "."):
            del sys.modules[name]


@pytest.fixture(autouse=True)
def _restore_filters_sysmodules_snapshot():
    """Snapshot ``sys.modules`` entries for the filters subgraph and restore
    them at teardown so a fresh import inside a test doesn't rebind the
    ``MRMR`` class object globally. Pre-fix this leaked across tests: any
    later cache-dependent test imported MRMR at module load (OLD class), but
    the lazy ``from .mrmr import MRMR`` inside ``_mrmr_fit_impl.py`` resolved
    to the rebound NEW class — cache writes landed on NEW, asserts on OLD.
    2026-05-22 trace identified this as the canonical pollution pattern.
    """
    snapshot = {
        name: mod for name, mod in sys.modules.items()
        if name == _FILTERS_MODULE or name.startswith(_FILTERS_MODULE + ".")
        or name == "mlframe.training.core._setup_helpers"
    }
    yield
    # Restore originals so module-level ``from ... import MRMR`` references
    # in other test files (and the lazy import inside _mrmr_fit_impl.py)
    # keep pointing at the same MRMR class instance after this test ends.
    # Iterate the union of snapshot keys and current keys: the ``use_mrmr_fs=False``
    # path deletes the filters subgraph and never re-imports it, so a current-keys-only
    # loop would never re-insert the snapshotted (OLD) module objects -- leaving filters
    # absent and forcing the NEXT test's import to bind a fresh NEW class (the identity
    # split that breaks pickling + MRMR._FIT_CACHE asserts in sibling tests).
    relevant = lambda name: (name == _FILTERS_MODULE or name.startswith(_FILTERS_MODULE + ".")
                             or name == "mlframe.training.core._setup_helpers")
    for name in set(sys.modules) | set(snapshot):
        if not relevant(name):
            continue
        if name in snapshot:
            sys.modules[name] = snapshot[name]
        elif name in sys.modules:
            del sys.modules[name]


def test_setup_helpers_module_has_no_top_level_mrmr_attribute():
    """Top-level ``MRMR`` re-export was the cheapest forensics for the eager
    import. Confirm it is gone (TYPE_CHECKING-guarded references are invisible
    at runtime so ``hasattr`` returns False).
    """
    from mlframe.training.core import _setup_helpers
    assert not hasattr(_setup_helpers, "MRMR"), (
        "_setup_helpers must NOT re-export MRMR at module level -- that would "
        "force the ~16s feature_selection.filters import on every caller"
    )


def test_filters_subgraph_not_loaded_after_setup_helpers_import():
    """After fresh import of _setup_helpers, the filters subgraph stays out
    of sys.modules until something explicitly asks for it.
    """
    _force_clean_mrmr_modules()
    # Force a fresh import of _setup_helpers itself so the deferral path is
    # exercised on this turn.
    sys.modules.pop("mlframe.training.core._setup_helpers", None)
    importlib.import_module("mlframe.training.core._setup_helpers")
    assert _FILTERS_MODULE not in sys.modules, (
        f"{_FILTERS_MODULE} ended up in sys.modules just from importing "
        f"_setup_helpers -- the deferral regressed"
    )


def test_build_pre_pipelines_with_use_mrmr_false_does_not_import_filters():
    """Opt-out path: caller passes ``use_mrmr_fs=False``. The MRMR import
    must NOT fire -- this is the whole point of the deferral.
    """
    _force_clean_mrmr_modules()
    sys.modules.pop("mlframe.training.core._setup_helpers", None)
    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
        use_ordinary_models=True,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        custom_pre_pipelines=None,
    )
    assert _FILTERS_MODULE not in sys.modules, (
        f"{_FILTERS_MODULE} loaded even though use_mrmr_fs=False -- the "
        f"deferral did not gate correctly"
    )
    # And the pipelines list should not contain MRMR
    assert all("MRMR" not in name for name in pre_pipeline_names)


def test_build_pre_pipelines_with_use_mrmr_true_does_import_filters():
    """Opt-in path: caller passes ``use_mrmr_fs=True``. The MRMR import
    MUST fire (otherwise the opt-in is silently broken).
    """
    _force_clean_mrmr_modules()
    sys.modules.pop("mlframe.training.core._setup_helpers", None)
    from mlframe.training.core._setup_helpers import _build_pre_pipelines

    pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
        use_ordinary_models=True,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=True,
        mrmr_kwargs={},
        custom_pre_pipelines=None,
    )
    assert _FILTERS_MODULE in sys.modules, (
        f"{_FILTERS_MODULE} NOT in sys.modules even though use_mrmr_fs=True -- "
        f"the deferral broke the opt-in path"
    )
    # And the pipelines list should contain MRMR
    assert any("MRMR" in name for name in pre_pipeline_names)
