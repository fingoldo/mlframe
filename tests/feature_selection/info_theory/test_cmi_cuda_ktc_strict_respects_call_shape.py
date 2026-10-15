"""Regression test for ``cmi_use_cuda``'s STRICT check (2026-07-11 fix).

``_cmi_cuda._should_use_cuda`` passes its OWN call shape (``n``, ``p``) to ``fe_gpu_strict_enabled`` so a
late-round call with a shrunk candidate count (small ``p``) correctly DECLINES STRICT even when the overall
fit is huge -- the per-call work floor (``n*p >= 1_000_000 and p >= 64``) exists specifically to stop STRICT
forcing a trivially small dispatch onto the GPU. When that call declines, ``_should_use_cuda`` falls through
to the KTC crossover, ``cmi_use_cuda(n, p)`` -- which used to call ``fe_gpu_strict_enabled()`` with NO
arguments. Since the floor is a no-op whenever ``n``/``p`` are omitted, this silently RE-APPROVED the exact
small-p call the caller had just correctly declined, defeating the floor entirely for any call that reached
this fallback under STRICT.

Fixed by passing ``n``/``p`` through here too, matching the caller's own usage.
"""

from __future__ import annotations

import pytest

import mlframe.feature_selection.filters._fe_gpu_strict as _strict_mod
from mlframe.feature_selection.filters.info_theory._cmi_cuda_ktc import cmi_use_cuda


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Helper that isolate."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT_AUTO_MIN_N", raising=False)
    # ``gpu_globally_disabled()`` (consulted downstream of the mocked ``_cuda_usable``) otherwise
    # silently overrides ``_cuda_usable``'s mock whenever the ambient CI/host env carries the off-switch --
    # see the identical fixture in test_batch_pair_mi_gpu_vram_guard.py for the full root-cause writeup.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    _strict_mod.clear_auto_fit_n()
    monkeypatch.setattr(_strict_mod, "_cuda_usable", lambda: True)
    yield
    _strict_mod.clear_auto_fit_n()


def test_strict_engages_small_p_call_under_force(monkeypatch):
    """Owner decision 2026-10-15: explicit STRICT forces CUDA on every call; the per-call p floor does not apply."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    assert cmi_use_cuda(n=1_000_000, p=2) is True


def test_strict_approves_large_call_under_force(monkeypatch):
    """Strict approves large call under force."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    assert cmi_use_cuda(n=1_000_000, p=64) is True


def test_strict_engages_small_np_product_under_force(monkeypatch):
    """Owner decision 2026-10-15: explicit STRICT ignores the n*p work floor as well."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    assert cmi_use_cuda(n=100, p=64) is True


def test_auto_mode_column_aware_gate_still_respects_call_shape(monkeypatch):
    """Same bug class under the AUTO (unset-env) column-aware gate from the companion fix: a huge/wide fit
    that auto-engages STRICT at the fit level must still decline a small-p individual call here."""
    _strict_mod.set_auto_fit_n(79_237, 544)  # engages the column-aware AUTO path
    assert _strict_mod.fe_gpu_strict_enabled() is True  # sanity: the fit-level gate is indeed on
    result = cmi_use_cuda(n=79_237, p=2)  # late-round remnant call
    assert result is not True


def test_strict_min_p_is_moot_under_force(monkeypatch):
    """Owner decision 2026-10-15: under explicit STRICT the shape floors, including the min_p override used by
    permutation-null callers, do not apply. Every shape engages."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    assert _strict_mod.fe_gpu_strict_enabled(n=250_000, p=6) is True
    assert _strict_mod.fe_gpu_strict_enabled(n=250_000, p=6, min_p=2) is True
    assert _strict_mod.fe_gpu_strict_enabled(n=1_000, p=12, min_p=2) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
