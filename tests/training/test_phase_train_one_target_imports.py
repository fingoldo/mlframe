"""Regression: names referenced inside core/ functions that were not imported after the refactor.

Module imports never trip because the names live inside function bodies (Python resolves lazily), so the bugs surface only at runtime in the
matching code paths. These tests fail fast at import time and verify .main's local-import path is not circular.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("module_path,name", [
    ("mlframe.training.core._phase_train_one_target", "score_ensemble"),
    ("mlframe.training.core._phase_train_one_target", "maybe_clean_ram_and_gpu"),
    ("mlframe.training.core._phase_train_one_target", "filter_existing"),
    ("mlframe.training.core._phase_train_one_target", "_filter_polars_cat_features_by_dtype"),
    ("mlframe.training.core._phase_train_one_target", "_format_temporal_audit_report"),
    ("mlframe.training.core._phase_train_one_target", "_plot_target_over_time"),
    ("mlframe.training.core._phase_helpers",          "os"),
    ("mlframe.training.core._phase_helpers",          "PreprocessingExtensionsConfig"),
    ("mlframe.training.core._misc_helpers",           "sys"),
    ("mlframe.training.core._setup_helpers",          "log_ram_usage"),
    ("mlframe.training.core.predict",                 "defaultdict"),
    ("mlframe.training.core.predict",                 "get_pandas_view_of_polars_df"),
    ("mlframe.training.core.predict",                 "stats"),
])
def test_module_level_name_resolves(module_path, name):
    mod = importlib.import_module(module_path)
    assert hasattr(mod, name), f"{name} missing from {module_path} module namespace"


def test_prep_polars_df_local_import_does_not_cycle():
    # CODE-P1-7: _prep_polars_df was hoisted to _misc_helpers.py so .main and ._phase_train_one_target
    # both import it from there at module top — no in-function local import remains in the hot loop.
    # Both modules must be importable in either order without ImportError.
    pt_mod = importlib.import_module("mlframe.training.core._phase_train_one_target")
    main_mod = importlib.import_module("mlframe.training.core.main")
    misc_mod = importlib.import_module("mlframe.training.core._misc_helpers")
    assert hasattr(misc_mod, "_prep_polars_df"), "_prep_polars_df must live in _misc_helpers"
    assert hasattr(pt_mod, "_prep_polars_df"), "_phase_train_one_target must re-expose via top-level import"
    # Back-compat: main.py still re-exports the symbol for downstream callers.
    assert hasattr(main_mod, "_prep_polars_df")


def test_prep_polars_df_no_local_import_in_train_one_target():
    """Regression for CODE-P1-7: ensure the hot loop does not perform `from .main import _prep_polars_df`."""
    import inspect

    from mlframe.training.core import _phase_train_one_target as pt

    src = inspect.getsource(pt)
    assert "from .main import _prep_polars_df" not in src, (
        "CODE-P1-7 regression: _prep_polars_df is being locally imported from .main inside the hot loop"
    )
