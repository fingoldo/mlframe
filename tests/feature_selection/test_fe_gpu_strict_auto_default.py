"""Size-gated AUTO default for MLFRAME_FE_GPU_STRICT.

STRICT GPU-resident FE is measured selection-equivalent to the CPU path once n is large (convergence by ~50k) and
~2.5x faster there. The three-state gate: explicit "1"/"0" force on/off; UNSET (or "auto") engages STRICT only when
the current fit's row count (set via ``set_auto_fit_n``) is at/above ``MLFRAME_FE_GPU_STRICT_AUTO_MIN_N`` AND a CUDA
device is usable. Small-n and no-GPU stay on the exact CPU path (byte-identical legacy).
"""
import pytest

import mlframe.feature_selection.filters._fe_gpu_strict as S


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT_AUTO_MIN_N", raising=False)
    S.clear_auto_fit_n()
    yield
    S.clear_auto_fit_n()


def _with_cuda(monkeypatch, present: bool):
    monkeypatch.setattr(S, "_cuda_usable", lambda: present)


def test_explicit_off_never_engages_even_large_n(monkeypatch):
    _with_cuda(monkeypatch, True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    S.set_auto_fit_n(1_000_000)
    assert S.fe_gpu_strict_enabled() is False


def test_explicit_on_engages_even_small_n_when_cuda(monkeypatch):
    _with_cuda(monkeypatch, True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    S.set_auto_fit_n(100)
    assert S.fe_gpu_strict_enabled() is True


def test_explicit_on_is_noop_without_cuda(monkeypatch):
    _with_cuda(monkeypatch, False)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    S.set_auto_fit_n(1_000_000)
    assert S.fe_gpu_strict_enabled() is False


def test_auto_off_below_threshold(monkeypatch):
    _with_cuda(monkeypatch, True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_AUTO_MIN_N", "50000")
    S.set_auto_fit_n(49_999)
    assert S.fe_gpu_strict_enabled() is False


def test_auto_on_at_threshold_with_cuda(monkeypatch):
    _with_cuda(monkeypatch, True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_AUTO_MIN_N", "50000")
    S.set_auto_fit_n(50_000)
    assert S.fe_gpu_strict_enabled() is True


def test_default_threshold_is_100k(monkeypatch):
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(60_000)
    assert S.fe_gpu_strict_enabled() is False  # 60k < 100k default -> AUTO stays off (no existing-test blast radius)
    S.set_auto_fit_n(100_000)
    assert S.fe_gpu_strict_enabled() is True


def test_auto_on_above_threshold_accepts_auto_literal(monkeypatch):
    _with_cuda(monkeypatch, True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "auto")
    S.set_auto_fit_n(120_000)
    assert S.fe_gpu_strict_enabled() is True


def test_auto_is_noop_without_cuda(monkeypatch):
    _with_cuda(monkeypatch, False)
    S.set_auto_fit_n(1_000_000)
    assert S.fe_gpu_strict_enabled() is False


def test_auto_off_when_no_fit_context(monkeypatch):
    _with_cuda(monkeypatch, True)
    S.clear_auto_fit_n()
    assert S.fe_gpu_strict_enabled() is False


def test_threshold_env_override(monkeypatch):
    _with_cuda(monkeypatch, True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_AUTO_MIN_N", "20000")
    S.set_auto_fit_n(25_000)
    assert S.fe_gpu_strict_enabled() is True
    S.set_auto_fit_n(19_999)
    assert S.fe_gpu_strict_enabled() is False


def test_clear_resets_context(monkeypatch):
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(1_000_000)
    assert S.fe_gpu_strict_enabled() is True
    S.clear_auto_fit_n()
    assert S.fe_gpu_strict_enabled() is False
