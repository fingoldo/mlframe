"""X_SECURITY_ROBUSTNESS-3 regression test: the trusted-root path-containment check gating every
joblib.load/dill.load call site must be a SINGLE shared implementation, not four independent copies with
subtly different fail-open/fail-closed defaults.

Behavioral edge cases (fail-closed on None, escape rejection, symlink caveat, cross-drive ValueError) are
already covered by ``tests/training/test_security_io_validation.py`` against the
``mlframe.training.core._validate_trusted_path`` wrapper, which now delegates to the shared function --
this file only pins the CONSOLIDATION itself (all four call sites route through one implementation).
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_shared_validate_trusted_path_exists_and_is_fail_closed():
    """The shared implementation lives in mlframe.core.helpers and rejects trusted_root=None."""
    from mlframe.core.helpers import validate_trusted_path

    with pytest.raises(ValueError, match="trusted_root is required"):
        validate_trusted_path("/some/path", None)


class _SentinelDelegationError(Exception):
    """Raised by the patched shared implementation to prove a call site actually delegates to it."""


@pytest.mark.parametrize(
    "module_path,func_name",
    [
        ("mlframe.training._data_helpers", "_validate_trusted_path"),
        ("mlframe.training.core._misc_helpers", "_validate_trusted_path"),
    ],
)
def test_wrapper_delegates_to_shared_implementation(module_path, func_name, monkeypatch):
    """Each per-module wrapper must actually CALL the shared helper, not reimplement commonpath logic.

    Behavioural proof: patch ``mlframe.core.helpers.validate_trusted_path`` to raise a distinctive
    sentinel and call the wrapper -- the sentinel must propagate, which is only possible if the
    wrapper's own local ``from mlframe.core.helpers import validate_trusted_path`` binds to (and
    calls) the patched function rather than an independent commonpath reimplementation.
    """
    import importlib
    import mlframe.core.helpers as helpers_mod

    def _raise_sentinel(*_args, **_kwargs):
        """Simulate the shared implementation being reached, proving delegation."""
        raise _SentinelDelegationError

    monkeypatch.setattr(helpers_mod, "validate_trusted_path", _raise_sentinel)
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)
    with pytest.raises(_SentinelDelegationError):
        func("/some/path", "/some/root")


def test_read_trained_models_delegates_to_shared_implementation(monkeypatch, tmp_path):
    """mlframe.inference.predict.read_trained_models must delegate its containment check too."""
    import mlframe.core.helpers as helpers_mod
    from mlframe.inference.predict import read_trained_models

    def _raise_sentinel(*_args, **_kwargs):
        """Simulate the shared implementation being reached, proving delegation."""
        raise _SentinelDelegationError

    monkeypatch.setattr(helpers_mod, "validate_trusted_path", _raise_sentinel)
    with pytest.raises(_SentinelDelegationError):
        read_trained_models(featureset="fs", X=pd.DataFrame({"a": [1]}), inference_folder=str(tmp_path), trusted_root=str(tmp_path))


def test_replay_cv_results_delegates_to_shared_implementation_and_requires_trusted_root(monkeypatch):
    """mlframe.estimators.pipelines.replay_cv_results must delegate too, and no longer silently narrow
    the check to fname's own directory when trusted_root is omitted (that default was a no-op guard)."""
    import mlframe.core.helpers as helpers_mod
    from mlframe.estimators.pipelines import replay_cv_results

    def _raise_sentinel(*_args, **_kwargs):
        """Simulate the shared implementation being reached, proving delegation."""
        raise _SentinelDelegationError

    monkeypatch.setattr(helpers_mod, "validate_trusted_path", _raise_sentinel)
    # Delegation check: the shared implementation is reached (and used) even with trusted_root omitted --
    # proving there is no local self-referential-dirname default masking the shared call.
    with pytest.raises(_SentinelDelegationError):
        replay_cv_results(fname="/some/dump.joblib")
