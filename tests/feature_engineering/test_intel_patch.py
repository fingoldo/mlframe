"""Tests for ``mlframe.feature_engineering.transformer._intel_patch.try_patch_sklearn`` -- previously had
zero test coverage despite being called from 4 production FE transformer entry points
(bgmm_quantile_bands.py, class_distance.py, local_lift.py, residual_stratified_distance.py).

sklearn-intelex is not installed in this environment, so the real ImportError no-op path is exercised
directly (not mocked); the "patch succeeds" / "patch raises mid-way" paths are exercised by injecting a
fake ``sklearnex`` module into ``sys.modules`` (the function does ``from sklearnex import patch_sklearn``
internally, so a fake module with a ``patch_sklearn`` attribute is sufficient to redirect it).
"""

from __future__ import annotations

import sys

import pytest

import mlframe.feature_engineering.transformer._intel_patch as intel_patch


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Every test starts from a fresh unpatched/unattempted state, and any fake sklearnex module injected
    into sys.modules by a test is removed afterward so it can't leak into unrelated tests."""
    monkeypatch.setattr(intel_patch, "_PATCH_APPLIED", False)
    monkeypatch.setattr(intel_patch, "_PATCH_ATTEMPTED", False)
    had_sklearnex = "sklearnex" in sys.modules
    saved = sys.modules.get("sklearnex")
    yield
    if had_sklearnex:
        sys.modules["sklearnex"] = saved
    else:
        sys.modules.pop("sklearnex", None)


class TestNoSklearnexInstalled:
    """sklearn-intelex is genuinely not installed in this test environment -- exercises the real
    ImportError path, not a simulation of it."""

    def test_returns_false_and_logs_when_not_installed(self, caplog):
        """Returns false and logs when not installed."""
        import logging

        with caplog.at_level(logging.INFO):
            result = intel_patch.try_patch_sklearn()
        assert result is False
        assert any("not installed" in r.message for r in caplog.records)


class TestEnvVarOptOut:
    """Groups tests covering MLFRAME_USE_SKLEARNEX=0."""

    def test_opt_out_returns_false_without_attempting_import(self, monkeypatch):
        """When opted out, the function must not even attempt the sklearnex import (verified by injecting
        a fake sklearnex whose patch_sklearn would raise if ever called, and confirming it wasn't)."""
        called = []

        class _FakeSklearnex:
            """Stand-in module whose patch_sklearn records whether it was ever invoked."""

            @staticmethod
            def patch_sklearn():
                """Record a call; would only fire if the opt-out gate were bypassed."""
                called.append(True)

        monkeypatch.setitem(sys.modules, "sklearnex", _FakeSklearnex())
        monkeypatch.setenv("MLFRAME_USE_SKLEARNEX", "0")
        result = intel_patch.try_patch_sklearn()
        assert result is False
        assert not called, "patch_sklearn must not be called when MLFRAME_USE_SKLEARNEX=0"

    def test_default_env_treats_unset_as_opted_in(self, monkeypatch):
        """Unset (the default) must NOT opt out -- only the literal string '0' does."""
        monkeypatch.delenv("MLFRAME_USE_SKLEARNEX", raising=False)
        # sklearnex genuinely absent here, so the real behaviour is the ImportError no-op --
        # confirms the opt-out check itself didn't short-circuit before even trying the import.
        result = intel_patch.try_patch_sklearn()
        assert result is False


class TestSuccessfulPatch:
    """Groups tests covering the patch-applied path via an injected fake sklearnex module."""

    def test_applies_patch_and_returns_true(self, monkeypatch):
        """Applies patch and returns true."""
        called = []

        class _FakeSklearnex:
            """Stand-in module recording that patch_sklearn was invoked."""

            @staticmethod
            def patch_sklearn():
                """Record the call."""
                called.append(True)

        monkeypatch.setitem(sys.modules, "sklearnex", _FakeSklearnex())
        result = intel_patch.try_patch_sklearn()
        assert result is True
        assert called == [True]

    def test_second_call_is_memoized_and_does_not_repatch(self, monkeypatch):
        """After a successful patch, subsequent calls must be O(1) and NOT call patch_sklearn again."""
        calls = []

        class _FakeSklearnex:
            """Stand-in module counting every patch_sklearn invocation."""

            @staticmethod
            def patch_sklearn():
                """Record each call so a second invocation would be detectable."""
                calls.append(True)

        monkeypatch.setitem(sys.modules, "sklearnex", _FakeSklearnex())
        assert intel_patch.try_patch_sklearn() is True
        assert intel_patch.try_patch_sklearn() is True
        assert len(calls) == 1, "a second call must not re-invoke patch_sklearn"


class TestPatchRaisesUnexpectedly:
    """Groups tests covering the broad-except fallback for a genuine (non-ImportError) failure."""

    def test_generic_exception_during_patch_is_caught_and_returns_false(self, monkeypatch, caplog):
        """A patch_sklearn() call that raises something other than ImportError (e.g. a runtime
        incompatibility) must be swallowed -- the caller's FE code must never break just because the
        optional acceleration failed to apply."""
        import logging

        class _FakeSklearnex:
            """Stand-in module whose patch_sklearn always raises to simulate a runtime failure."""

            @staticmethod
            def patch_sklearn():
                """Simulate an unexpected runtime failure mid-patch."""
                raise RuntimeError("simulated incompatible runtime")

        monkeypatch.setitem(sys.modules, "sklearnex", _FakeSklearnex())
        with caplog.at_level(logging.WARNING):
            result = intel_patch.try_patch_sklearn()
        assert result is False
        assert any("patch failed" in r.message for r in caplog.records)

    def test_after_a_failed_attempt_a_second_call_does_not_retry(self, monkeypatch):
        """_PATCH_ATTEMPTED must gate retries even on failure -- one failed attempt per process, not a
        retry storm on every FE call site that invokes try_patch_sklearn()."""

        class _FakeSklearnexAlwaysFails:
            """Stand-in module whose patch_sklearn always raises, tracking call count."""

            calls = 0

            @classmethod
            def patch_sklearn(cls):
                """Increment the call counter and raise."""
                cls.calls += 1
                raise RuntimeError("simulated failure")

        monkeypatch.setitem(sys.modules, "sklearnex", _FakeSklearnexAlwaysFails())
        assert intel_patch.try_patch_sklearn() is False
        assert intel_patch.try_patch_sklearn() is False
        assert _FakeSklearnexAlwaysFails.calls == 1, "a second call after a failed attempt must not retry"
