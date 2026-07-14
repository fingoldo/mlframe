"""Wave 12: zero-arg ``*_gpu_enabled()``/``*_use_*()`` wrappers now forward their caller's own (n, p) shape
to the canonical ``fe_gpu_strict_enabled(*, n=None, p=None)`` gate instead of calling it bare.

Before this fix, every wrapper below called ``fe_gpu_strict_enabled()`` with NO arguments: under explicit
``MLFRAME_FE_GPU_STRICT=1`` (or a large-fit AUTO context), ``_passes_call_work_floor()`` treats a missing
``n``/``p`` as "no per-call shape given -> pass" (see ``_fe_gpu_strict.py``'s ``_passes_call_work_floor``), so
the decision was IDENTICAL regardless of the actual candidate-matrix shape at that call site -- a wrapper
gating a single tiny 2-column call and one gating a 500-column batch returned the exact same answer. These
tests would have FAILED pre-fix (both the "tiny" and "large" call would return True under force, or both
would ignore the per-call floor identically); post-fix each wrapper's decision genuinely depends on the shape
threaded through to it, mirroring ``test_fe_gpu_strict_auto_default.py``'s direct gate-level pin.

Gating decisions only -- never the computed feature values / MI scores (those stay backend-selected but
bit-identical per backend; see the per-kernel GPU/CPU parity suites)."""
import pytest

import mlframe.feature_selection.filters._fe_gpu_strict as _S


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Clear all FE-GPU-strict env vars and the auto-fit-n cache before and after each test."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.delenv("MLFRAME_CMI_GPU", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GPU_DISCRETIZE", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GPU_BINNING", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GATE_RESIDENT_CANDS", raising=False)
    _S.clear_auto_fit_n()
    yield
    _S.clear_auto_fit_n()


def _force_strict_with_cuda(monkeypatch):
    """Force fe_gpu_strict_enabled's CUDA-usable + MLFRAME_FE_GPU_STRICT=1 path for this test."""
    monkeypatch.setattr(_S, "_cuda_usable", lambda: True)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")


# Below the per-call work floor (n*p < 1_000_000 or p < 64) vs comfortably above it.
_TINY = dict(n=100, p=2)
_BIG = dict(n=1_000_000, p=64)


def test_cmi_gpu_enabled_forwards_shape(monkeypatch):
    """_cmi_gpu_enabled gates on the caller's own (n, p) shape, not a bare shape-blind call."""
    _force_strict_with_cuda(monkeypatch)
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_gpu_enabled

    assert _cmi_gpu_enabled(**_TINY) is False
    assert _cmi_gpu_enabled(**_BIG) is True
    # Omitting shape entirely preserves the legacy fit-level-only (shape-blind) behavior.
    assert _cmi_gpu_enabled() is True


def test_orth_mi_gpu_enabled_forwards_shape(monkeypatch):
    """_orth_mi_gpu_enabled gates on the caller's own (n, p) shape."""
    _force_strict_with_cuda(monkeypatch)
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import _orth_mi_gpu_enabled

    assert _orth_mi_gpu_enabled(**_TINY) is False
    assert _orth_mi_gpu_enabled(**_BIG) is True
    assert _orth_mi_gpu_enabled() is True


def test_binnedmi_gpu_enabled_forwards_shape(monkeypatch):
    """_binnedmi_gpu_enabled gates on the caller's own (n, p) shape."""
    _force_strict_with_cuda(monkeypatch)
    from mlframe.feature_selection.filters._wavelet_basis_fe import _binnedmi_gpu_enabled

    assert _binnedmi_gpu_enabled(**_TINY) is False
    assert _binnedmi_gpu_enabled(**_BIG) is True
    assert _binnedmi_gpu_enabled() is True


def test_pair_gate_resident_enabled_forwards_shape(monkeypatch):
    """_pair_gate_resident_enabled gates on the caller's own (n, p) shape."""
    _force_strict_with_cuda(monkeypatch)
    from mlframe.feature_selection.filters._mrmr_fe_step._step_pairs_rank import _pair_gate_resident_enabled

    assert _pair_gate_resident_enabled(**_TINY) is False
    assert _pair_gate_resident_enabled(**_BIG) is True
    assert _pair_gate_resident_enabled() is True


def test_permnull_use_resident_forwards_shape(monkeypatch):
    """permnull_use_resident gates on the caller's own (n, ncand) shape."""
    _force_strict_with_cuda(monkeypatch)
    from mlframe.feature_selection.filters._permutation_null_resident_ktc import permnull_use_resident

    assert permnull_use_resident(n=100, ncand=2, nperm=25) is False
    assert permnull_use_resident(n=1_000_000, ncand=64, nperm=25) is True


def test_rescand_use_resident_forwards_shape(monkeypatch):
    """rescand_use_resident gates on the caller's own (n, k) shape."""
    _force_strict_with_cuda(monkeypatch)
    from mlframe.feature_selection.filters._resident_candidate_mi_ktc import rescand_use_resident

    assert rescand_use_resident(n=100, k=2) is False
    assert rescand_use_resident(n=1_000_000, k=64) is True


def test_pool_table_use_resident_forwards_shape(monkeypatch):
    """pool_table_use_resident gates on the caller's own (n_rows, npairs, n_combos) shape."""
    _force_strict_with_cuda(monkeypatch)
    from mlframe.feature_selection.filters._usability_pool_resident_ktc import pool_table_use_resident

    assert pool_table_use_resident(n_rows=100, npairs=2, n_combos=8) is False
    assert pool_table_use_resident(n_rows=1_000_000, npairs=64, n_combos=8) is True


def test_shufflegen_use_gpu_forwards_shape(monkeypatch):
    """shufflegen_use_gpu forwards n (no natural per-call column count) to the shape-aware gate."""
    _force_strict_with_cuda(monkeypatch)
    from mlframe.feature_selection.filters._permutation_null_shufflegen_ktc import shufflegen_use_gpu

    # shufflegen_use_gpu forwards only n (no natural per-call column count) -- still shape-aware on n alone
    # since fe_gpu_strict_enabled(n=..., p=None) keeps the per-call floor a no-op (p missing -> "pass"), so
    # this pins that a real p=64+ call clears while distinguishing from the bare (shape-blind) call is covered
    # by the p-bearing wrappers above; here we confirm the call accepts/forwards n without raising.
    assert shufflegen_use_gpu(n=1_000_000, nperm=25) is True


def test_fe_gpu_discretize_and_binning_enabled_forward_shape(monkeypatch):
    """_fe_gpu_discretize_enabled / _fe_gpu_binning_enabled both gate on the caller's own (n_rows, n_cands) shape."""
    _force_strict_with_cuda(monkeypatch)
    import pyutilz.core.pythonlib as _pl
    monkeypatch.setattr(_pl, "is_cuda_available", lambda: True)
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import (
        _fe_gpu_discretize_enabled, _fe_gpu_binning_enabled,
    )

    assert _fe_gpu_discretize_enabled(n_rows=100, n_cands=2) is False
    assert _fe_gpu_discretize_enabled(n_rows=1_000_000, n_cands=64) is True
    assert _fe_gpu_binning_enabled(n_rows=100, n_cands=2) is False
    assert _fe_gpu_binning_enabled(n_rows=1_000_000, n_cands=64) is True
