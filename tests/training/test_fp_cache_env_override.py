"""Regression sensor for A5 Low #14.

Pre-fix _FP_CACHE_MAX was a hard-coded module global; long-running Jupyter sessions that looped many suites with rotating frame ids had no way to retune without editing source. The fix reads ``MLFRAME_FP_CACHE_MAX`` via the ``_fp_cache_max_default`` helper; invalid / non-positive values fall back to the 128 default.

We deliberately test the helper directly (not via sys.modules reload) because reloading the fingerprint module would rebind module objects whose class identity is depended on by other tests (per CLAUDE.md test-pollution rule + memory `feedback_no_module_reload_without_snapshot`).
"""

from __future__ import annotations


from mlframe.training.feature_handling.fingerprint import _fp_cache_max_default


def test_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("MLFRAME_FP_CACHE_MAX", raising=False)
    assert _fp_cache_max_default() == 128


def test_env_override_takes_effect(monkeypatch):
    monkeypatch.setenv("MLFRAME_FP_CACHE_MAX", "512")
    assert _fp_cache_max_default() == 512


def test_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MLFRAME_FP_CACHE_MAX", "not-a-number")
    assert _fp_cache_max_default() == 128


def test_zero_or_negative_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MLFRAME_FP_CACHE_MAX", "0")
    assert _fp_cache_max_default() == 128
    monkeypatch.setenv("MLFRAME_FP_CACHE_MAX", "-5")
    assert _fp_cache_max_default() == 128
