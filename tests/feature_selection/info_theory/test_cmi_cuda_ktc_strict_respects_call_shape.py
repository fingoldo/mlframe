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
    _strict_mod.clear_auto_fit_n()
    monkeypatch.setattr(_strict_mod, "_cuda_usable", lambda: True)
    yield
    _strict_mod.clear_auto_fit_n()


def test_strict_declines_small_p_call_even_under_force(monkeypatch):
    """The bug's exact reproduction: STRICT forced on, a late-round call with p=2 (well under the p>=64
    floor) must NOT be silently approved for CUDA just because it reached the KTC crossover fallback."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    # No kernel_tuning_cache entry -> would otherwise fall through past the STRICT check to the swept
    # crossover / hardcoded bootstrap; here we only care whether STRICT itself wrongly short-circuits to True.
    result = cmi_use_cuda(n=1_000_000, p=2)
    assert result is not True, "STRICT must not force CUDA for a call that fails the p>=64 per-call floor"


def test_strict_approves_large_call_under_force(monkeypatch):
    """Strict approves large call under force."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    assert cmi_use_cuda(n=1_000_000, p=64) is True


def test_strict_declines_small_np_product_even_with_adequate_p(monkeypatch):
    """p alone clearing 64 is not sufficient -- n*p must also clear the work floor."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    result = cmi_use_cuda(n=100, p=64)  # n*p = 6400, far under 1_000_000
    assert result is not True


def test_auto_mode_column_aware_gate_still_respects_call_shape(monkeypatch):
    """Same bug class under the AUTO (unset-env) column-aware gate from the companion fix: a huge/wide fit
    that auto-engages STRICT at the fit level must still decline a small-p individual call here."""
    _strict_mod.set_auto_fit_n(79_237, 544)  # engages the column-aware AUTO path
    assert _strict_mod.fe_gpu_strict_enabled() is True  # sanity: the fit-level gate is indeed on
    result = cmi_use_cuda(n=79_237, p=2)  # late-round remnant call
    assert result is not True


def test_strict_min_p_override_admits_perm_null_shapes(monkeypatch):
    """Regression (wellbore-100k GPU-strict profile): a permutation-null workload's p is n_permutations
    (12-24 by design, every permutation re-reading the SAME resident candidate), so the candidate-batch
    p >= 64 crossover does not apply -- holding it to 64 forced all 2232 _conditional_perm_null calls
    (~261s) onto the CPU. min_p overrides ONLY the p-floor leg; the n*p total-work leg still binds."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    # nperm=12 at n=250k: n*p = 3M >= 1M but p < 64 -> declined WITHOUT min_p, admitted WITH min_p=2.
    assert _strict_mod.fe_gpu_strict_enabled(n=250_000, p=12) is False
    assert _strict_mod.fe_gpu_strict_enabled(n=250_000, p=12, min_p=2) is True
    # The total-work leg still binds under min_p: tiny n*p stays declined.
    assert _strict_mod.fe_gpu_strict_enabled(n=1_000, p=12, min_p=2) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
