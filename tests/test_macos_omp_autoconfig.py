"""``mlframe._autoconfigure_macos_omp_threads`` sets OMP_NUM_THREADS/KMP_DUPLICATE_LIB_OK
on darwin only, never overrides an explicit operator setting, and honours its opt-out.

See audits/ci_review_2026-09-08/_TRACKER.md, X5: LightGBM's own n_jobs=1 does not gate
every internal OpenMP call site on macOS (round 9 confirmed the crash still reproduced),
so the real OS env var has to be set as early as possible -- at ``import mlframe`` time.
"""

from __future__ import annotations

import os
import sys

import mlframe


def _clear_omp_env(monkeypatch):
    """Remove the vars this autoconfig touches so each test starts from a clean slate."""
    for k in ("OMP_NUM_THREADS", "KMP_DUPLICATE_LIB_OK", "MLFRAME_NO_MACOS_OMP_AUTOCONFIG"):
        monkeypatch.delenv(k, raising=False)


def test_sets_omp_env_on_darwin(monkeypatch):
    """On darwin with a clean env, both vars are set to their crash-avoiding values."""
    _clear_omp_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    mlframe._autoconfigure_macos_omp_threads()
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"


def test_noop_on_non_darwin(monkeypatch):
    """On linux/win32 the function returns immediately and touches nothing."""
    _clear_omp_env(monkeypatch)
    for plat in ("linux", "win32"):
        monkeypatch.setattr(sys, "platform", plat)
        mlframe._autoconfigure_macos_omp_threads()
        assert "OMP_NUM_THREADS" not in os.environ
        assert "KMP_DUPLICATE_LIB_OK" not in os.environ


def test_never_overrides_explicit_operator_setting(monkeypatch):
    """An operator who has already set OMP_NUM_THREADS is not silently overridden."""
    _clear_omp_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    mlframe._autoconfigure_macos_omp_threads()
    assert os.environ.get("OMP_NUM_THREADS") == "8"
    # KMP_DUPLICATE_LIB_OK was untouched by the operator, so it's still set.
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"


def test_opt_out_env_var_skips_entirely(monkeypatch):
    """MLFRAME_NO_MACOS_OMP_AUTOCONFIG=1 skips both vars, e.g. to test a fixed libomp release."""
    _clear_omp_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("MLFRAME_NO_MACOS_OMP_AUTOCONFIG", "1")
    mlframe._autoconfigure_macos_omp_threads()
    assert "OMP_NUM_THREADS" not in os.environ
    assert "KMP_DUPLICATE_LIB_OK" not in os.environ
