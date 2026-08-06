"""X_SECURITY_ROBUSTNESS-3 regression test: the trusted-root path-containment check gating every
joblib.load/dill.load call site must be a SINGLE shared implementation, not four independent copies with
subtly different fail-open/fail-closed defaults.

Behavioral edge cases (fail-closed on None, escape rejection, symlink caveat, cross-drive ValueError) are
already covered by ``tests/training/test_security_io_validation.py`` against the
``mlframe.training.core._validate_trusted_path`` wrapper, which now delegates to the shared function --
this file only pins the CONSOLIDATION itself (all four call sites route through one implementation).
"""

from __future__ import annotations

import inspect

import pytest


def test_shared_validate_trusted_path_exists_and_is_fail_closed():
    """The shared implementation lives in mlframe.core.helpers and rejects trusted_root=None."""
    from mlframe.core.helpers import validate_trusted_path

    with pytest.raises(ValueError, match="trusted_root is required"):
        validate_trusted_path("/some/path", None)


@pytest.mark.parametrize(
    "module_path,func_name",
    [
        ("mlframe.training._data_helpers", "_validate_trusted_path"),
        ("mlframe.training.core._misc_helpers", "_validate_trusted_path"),
    ],
)
def test_wrapper_delegates_to_shared_implementation(module_path, func_name):
    """Each per-module wrapper's source must call the shared helper, not reimplement commonpath logic."""
    import importlib

    mod = importlib.import_module(module_path)
    src = inspect.getsource(getattr(mod, func_name))
    assert "validate_trusted_path" in src, f"{module_path}.{func_name} no longer delegates to the shared implementation"
    assert ".commonpath(" not in src, f"{module_path}.{func_name} reimplements the commonpath check instead of delegating"


def test_read_trained_models_delegates_to_shared_implementation():
    """mlframe.inference.predict.read_trained_models must delegate its containment check too."""
    from mlframe.inference.predict import read_trained_models

    src = inspect.getsource(read_trained_models)
    assert "validate_trusted_path" in src
    assert ".commonpath(" not in src


def test_replay_cv_results_delegates_to_shared_implementation_and_requires_trusted_root():
    """mlframe.estimators.pipelines.replay_cv_results must delegate too, and no longer silently narrow
    the check to fname's own directory when trusted_root is omitted (that default was a no-op guard)."""
    from mlframe.estimators.pipelines import replay_cv_results

    src = inspect.getsource(replay_cv_results)
    assert "validate_trusted_path" in src
    assert ".commonpath(" not in src
    assert "os.path.dirname(abs_fname)" not in src, "the weak self-referential default must be gone"
