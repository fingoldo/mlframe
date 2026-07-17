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


# --------------------------------------------------------------------------------------------------
# Per-call shape gate (2026-07-09 fix). Before this, AUTO forced EVERY dispatch to STRICT for the
# rest of a >=100k-row fit, including trivially small late-round calls (e.g. p=2 remaining candidates)
# -- exactly the shape the per-host KTC crossover exists to route back to CPU. Passing n/p additionally
# requires n*p >= _STRICT_MIN_CALL_WORK before STRICT engages for THAT call.
# --------------------------------------------------------------------------------------------------


def test_auto_declines_tiny_call_even_when_fit_is_huge(monkeypatch):
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(4_000_000)  # a huge fit -- fit-level gate alone would engage STRICT
    # THIS call is a late-round remnant: p=2 candidates left, regardless of how large n is (the p floor
    # alone must decline it -- a huge n cannot compensate for a trivially small candidate count).
    assert S.fe_gpu_strict_enabled(n=4_000_000, p=2) is False
    # Also decline a moderate-p / small-n combination that fails only the WORK floor.
    assert S.fe_gpu_strict_enabled(n=100, p=100) is False


def test_auto_engages_large_call_on_huge_fit(monkeypatch):
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(4_000_000)
    assert S.fe_gpu_strict_enabled(n=4_000_000, p=64) is True  # n*p >= _STRICT_MIN_CALL_WORK


def test_no_shape_args_preserves_legacy_fit_level_only_gate(monkeypatch):
    """Callers that have not been updated to pass shape keep the exact pre-fix behavior."""
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(4_000_000)
    assert S.fe_gpu_strict_enabled() is True


def test_explicit_force_still_respects_call_work_floor(monkeypatch):
    _with_cuda(monkeypatch, True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    S.set_auto_fit_n(100)  # irrelevant under explicit force, but the call-shape floor still applies
    assert S.fe_gpu_strict_enabled(n=1000, p=2) is False
    assert S.fe_gpu_strict_enabled(n=1_000_000, p=64) is True


def test_call_work_floor_boundary_matches_constant(monkeypatch):
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(4_000_000)
    n, p = 1000, S._STRICT_MIN_CALL_WORK // 1000  # n*p exactly at the floor
    assert n * p == S._STRICT_MIN_CALL_WORK
    assert S.fe_gpu_strict_enabled(n=n, p=p) is True
    assert S.fe_gpu_strict_enabled(n=n, p=p - 1) is False


# --------------------------------------------------------------------------------------------------
# Column-aware fit-level AUTO gate (2026-07-11 fix). The row-only _auto_min_n() threshold (100k) ignored
# column count entirely -- a wide-but-under-100k-rows fit (real production shape: 79,237 rows x 544 cols,
# ~43M total elements) stayed on the exact CPU path purely for sitting under an arbitrary row mark, despite
# clearing the SAME n*p/p work floor a per-call dispatch would need by 40x+. set_auto_fit_n now accepts an
# optional column count; the AUTO gate engages when EITHER the pure row-count rule holds (unchanged) OR the
# fit is at/above the validated ~50k convergence floor AND its total (n, p) work clears the per-call floor.
# --------------------------------------------------------------------------------------------------


def test_wide_fit_under_row_threshold_now_engages_via_column_aware_gate(monkeypatch):
    """The motivating case: real production shape (79237 rows x 544 cols) is under the 100k row default but
    clears the column-aware alternate path (n>=50k, n*p>=1M, p>=64)."""
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(79_237, 544)
    assert S.fe_gpu_strict_enabled() is True


def test_narrow_fit_under_row_threshold_still_declines(monkeypatch):
    """A fit with few columns must NOT be swept in just because n is large-ish -- the p>=64 floor still
    applies to the fit-level decision, same as it always has for per-call decisions."""
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(79_237, 10)  # n*p = 792,370 < 1M AND p < 64 -- fails both column-aware conditions
    assert S.fe_gpu_strict_enabled() is False


def test_wide_fit_below_convergence_floor_still_declines():
    """Column count cannot compensate for too FEW rows -- the ~50k convergence floor is a hard row-count
    requirement regardless of how wide the fit is (MI-estimation convergence depends on n, not p)."""
    S.set_auto_fit_n(10_000, 10_000)  # n*p = 100M, comfortably clears the work floor, but n < 50k
    assert S.fe_gpu_strict_enabled() is False


def test_missing_column_count_falls_back_to_row_only_rule(monkeypatch):
    """A caller that only ever tracked row count (the pre-fix contract) must get EXACTLY the old behavior --
    never MORE eager just because column data happens to be unavailable."""
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(79_237)  # p omitted entirely
    assert S.fe_gpu_strict_enabled() is False  # would engage at (79237, 544) with p given, but not without it


def test_row_only_rule_at_100k_still_engages_regardless_of_column_count(monkeypatch):
    """The original n>=100k rule is UNCHANGED and must still fire even for a narrow (few-column) fit --
    this fix only ADDS an alternate path, it never narrows the existing one."""
    _with_cuda(monkeypatch, True)
    S.set_auto_fit_n(100_000, 2)  # narrow: only 2 columns, would fail the column-aware path outright
    assert S.fe_gpu_strict_enabled() is True


def test_column_aware_gate_is_noop_without_cuda(monkeypatch):
    _with_cuda(monkeypatch, False)
    S.set_auto_fit_n(79_237, 544)
    assert S.fe_gpu_strict_enabled() is False


def test_column_aware_gate_respects_explicit_off(monkeypatch):
    _with_cuda(monkeypatch, True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    S.set_auto_fit_n(79_237, 544)
    assert S.fe_gpu_strict_enabled() is False
