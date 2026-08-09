"""Regression: names referenced inside core/ functions that were not imported after the refactor.

Module imports never trip because the names live inside function bodies (Python resolves lazily), so the bugs surface only at runtime in the
matching code paths. These tests fail fast at import time and verify .main's local-import path is not circular.

Module paths below were re-pointed 2026-08 (CI caught the drift on shard 8/8, all 5 required Python versions):
a LATER monolith split carved ``_phase_train_one_target.py`` into per-concern siblings
(``_phase_train_one_target_ensembling.py``, ``_phase_train_one_target_dataset_cache.py``,
``_phase_train_one_target_model_setup.py``, ``_phase_train_one_target_body.py``,
``_phase_train_one_target_polars_fastpath.py``) -- each call site's own import is correct (verified: no
NameError risk at runtime), but this test's parametrize list still pointed at the pre-split facade module,
which never needed these names in its own namespace post-split. Re-pointed each entry at the sibling that
actually calls the name. ``predict.py``'s ``stats`` entry was dropped outright: no ``stats`` name (bare or
``scipy.stats``) is referenced anywhere in ``predict.py`` or its split-out siblings -- dead assertion, not a
stale pointer.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path,name",
    [
        ("mlframe.training.core._phase_train_one_target_ensembling", "score_ensemble"),
        ("mlframe.training.core._phase_train_one_target_dataset_cache", "maybe_clean_ram_and_gpu"),
        ("mlframe.training.core._phase_train_one_target_body", "filter_existing"),
        ("mlframe.training.core._phase_train_one_target_body", "_filter_polars_cat_features_by_dtype"),
        ("mlframe.training.core._phase_train_one_target_model_setup", "_format_temporal_audit_report"),
        ("mlframe.training.core._phase_train_one_target_model_setup", "_plot_target_over_time"),
        ("mlframe.training.core._phase_helpers", "os"),
        ("mlframe.training.core._phase_helpers_fit_pipeline", "PreprocessingExtensionsConfig"),
        ("mlframe.training.core._misc_helpers", "sys"),
        ("mlframe.training.core._setup_helpers_outliers", "log_ram_usage"),
        ("mlframe.training.core.predict", "defaultdict"),
        ("mlframe.training.core.predict", "get_pandas_view_of_polars_df"),
    ],
)
def test_module_level_name_resolves(module_path, name):
    """Each (module, name) pair in the parametrize matrix must resolve as a real module-level attribute."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, name), f"{name} missing from {module_path} module namespace"


def test_prep_polars_df_local_import_does_not_cycle():
    """_prep_polars_df lives in _misc_helpers.py; main.py and _phase_train_one_target_polars_fastpath.py
    (the sibling that actually calls it post-split -- see the module docstring above) both import it
    top-level without a cycle."""
    # CODE-P1-7: _prep_polars_df was hoisted to _misc_helpers.py so .main and the polars-fastpath sibling
    # both import it from there at module top — no in-function local import remains in the hot loop.
    # Both modules must be importable in either order without ImportError.
    fastpath_mod = importlib.import_module("mlframe.training.core._phase_train_one_target_polars_fastpath")
    main_mod = importlib.import_module("mlframe.training.core.main")
    misc_mod = importlib.import_module("mlframe.training.core._misc_helpers")
    assert hasattr(misc_mod, "_prep_polars_df"), "_prep_polars_df must live in _misc_helpers"
    assert hasattr(fastpath_mod, "_prep_polars_df"), "the polars-fastpath sibling must re-expose via top-level import"
    # Back-compat: main.py still re-exports the symbol for downstream callers.
    assert hasattr(main_mod, "_prep_polars_df")


def test_prep_polars_df_top_level_binding_is_canonical():
    """Regression for CODE-P1-7: pre-fix the hot loop did ``from .main import _prep_polars_df``,
    which (a) created a cyclic-import smell and (b) cost a sys.modules lookup per call. The fix
    hoists the symbol to _misc_helpers and imports it once at module top in the polars-fastpath
    sibling (the actual caller post-split -- see the module docstring above). Behavioural assertion:
    the module-level binding there must be the SAME object as the canonical _misc_helpers symbol
    (identity check), proving no in-function rebinding has displaced it."""
    from mlframe.training.core import _phase_train_one_target_polars_fastpath as fastpath
    from mlframe.training.core import _misc_helpers as misc

    assert fastpath._prep_polars_df is misc._prep_polars_df, (
        "CODE-P1-7 regression: fastpath._prep_polars_df is not the canonical _misc_helpers symbol; "
        "a stale rebinding (e.g. in-function import) likely shadows the top-level reference."
    )
