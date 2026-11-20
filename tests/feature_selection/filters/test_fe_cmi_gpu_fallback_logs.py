"""The CMI null / gate / qbin / device-enumeration fallbacks log a WARNING (driven on CPU by stubbing the GPU call to fail), so the slower path is never silent."""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the per-key log counts so each test sees its first occurrence."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _warned(caplog, text):
    """Whether a WARNING record contains ``text``."""
    return any(r.levelno >= logging.WARNING and text in r.getMessage() for r in caplog.records)


def _boom(*args, **kwargs):
    """Fail like a device fault."""
    raise RuntimeError("device fault")


def _inputs(n=64):
    """A small host-only null-inputs object and its stratified-support stand-in."""
    from mlframe.feature_selection.filters import _fe_cmi_redundancy_null as mod

    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, n).astype(np.int64)
    y = rng.integers(0, 2, n).astype(np.int64)
    z = rng.integers(0, 4, n).astype(np.int64)
    order = np.argsort(z, kind="stable")
    sorted_z = z[order]
    z_rank = np.zeros(n)
    z_rank[1:] = np.cumsum(sorted_z[1:] != sorted_z[:-1])
    sup = SimpleNamespace(y_i=y, z_i=z, z=z, order=order, z_rank=z_rank)
    return mod, mod._NullInputs(x, y, 0, 0), sup


def _gpu_on(monkeypatch, value=True):
    """Make the CMI GPU gate report ``value``."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe

    monkeypatch.setattr(_mi_greedy_cmi_fe, "_cmi_gpu_enabled", lambda *a, **k: value)


def test_analytic_null_failure_warns(monkeypatch, caplog):
    """A failing analytic null returns None (the permutation null runs) and warns."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe

    mod, inp, _ = _inputs()
    monkeypatch.setenv("MLFRAME_CMI_ANALYTIC_NULL_MIN_N", "1")
    monkeypatch.setattr(_mi_greedy_cmi_fe, "_entropy_from_classes", _boom)
    with caplog.at_level(logging.WARNING):
        assert mod._analytic_cmi_null(inp, None, None, None, 0.95) is None
    assert _warned(caplog, "analytic CMI null failed")


def test_resident_support_null_failure_warns(monkeypatch, caplog):
    """A failing resident-support GPU null returns None and warns."""
    from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu

    mod, inp, _ = _inputs()
    _gpu_on(monkeypatch)
    monkeypatch.setattr(_fe_cmi_perm_null_gpu, "perm_null_gpu_resident_enabled", lambda: True)
    monkeypatch.setattr(_fe_cmi_perm_null_gpu, "conditional_perm_null_gpu", _boom)
    with caplog.at_level(logging.WARNING):
        assert mod._resident_support_null(inp, object(), 5, 0.95, 0, 0) is None
    assert _warned(caplog, "resident-support GPU conditional perm-null failed")


def test_marginal_gpu_null_failure_warns(monkeypatch, caplog):
    """A failing GPU marginal null warns before the host path runs."""
    from mlframe.feature_selection.filters import _fe_batched_mi, _fe_cmi_perm_null_gpu

    mod, inp, _ = _inputs()
    _gpu_on(monkeypatch)
    monkeypatch.setattr(mod, "_gpu_resident_flag", lambda: True)
    monkeypatch.setattr(_fe_cmi_perm_null_gpu, "conditional_perm_null_gpu", _boom)
    monkeypatch.setattr(_fe_batched_mi, "batched_cmi_gpu", _boom)
    with caplog.at_level(logging.WARNING):
        mod._marginal_perm_null(inp, 5, 0.95, 0, 0)
    assert _warned(caplog, "GPU-resident marginal perm-null failed")


def test_marginal_batched_null_failure_warns(monkeypatch, caplog):
    """A failing batched device marginal null warns before the host loop runs."""
    from mlframe.feature_selection.filters import _fe_batched_mi

    mod, inp, _ = _inputs()
    _gpu_on(monkeypatch)
    monkeypatch.setattr(mod, "_gpu_resident_flag", lambda: False)
    monkeypatch.setattr(_fe_batched_mi, "batched_cmi_gpu", _boom)
    with caplog.at_level(logging.WARNING):
        mod._marginal_perm_null(inp, 5, 0.95, 0, 0)
    assert _warned(caplog, "batched device marginal perm-null failed")


def test_conditional_gpu_null_failure_warns(monkeypatch, caplog):
    """A failing GPU-resident conditional null returns None and warns."""
    from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu

    mod, inp, sup = _inputs()
    _gpu_on(monkeypatch)
    monkeypatch.setattr(mod, "_gpu_resident_flag", lambda: True)
    monkeypatch.setattr(_fe_cmi_perm_null_gpu, "conditional_perm_null_gpu", _boom)
    with caplog.at_level(logging.WARNING):
        assert mod._conditional_gpu_resident_null(inp, sup, 5, 0.95, 0, 0) is None
    assert _warned(caplog, "GPU-resident conditional perm-null failed")


def test_conditional_batched_null_vram_refusal_warns(monkeypatch, caplog):
    """A VRAM-cushion refusal of the batched conditional null returns None and warns."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    mod, inp, sup = _inputs()
    _gpu_on(monkeypatch)
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *a, **k: False)
    with caplog.at_level(logging.WARNING):
        assert mod._conditional_batched_gpu_null(inp, sup, 5, 0.95) is None
    assert _warned(caplog, "refused by the VRAM cushion")


def test_conditional_batched_null_failure_warns(monkeypatch, caplog):
    """A failing batched conditional null returns None and warns."""
    from mlframe.feature_selection.filters import _fe_gpu_vram

    mod, inp, sup = _inputs()
    _gpu_on(monkeypatch)
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *a, **k: True)
    with caplog.at_level(logging.WARNING):
        assert mod._conditional_batched_gpu_null(inp, sup, 5, 0.95) is None
    assert _warned(caplog, "batched device conditional null failed")


def test_swept_device_null_failure_warns(monkeypatch, caplog):
    """A failing swept device null returns None (the host table null runs) and warns."""
    from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu

    mod, inp, sup = _inputs()
    _gpu_on(monkeypatch, False)
    monkeypatch.setattr(_fe_cmi_perm_null_gpu, "perm_null_on_device", lambda n: True)
    monkeypatch.setattr(_fe_cmi_perm_null_gpu, "conditional_perm_null_gpu", _boom)
    with caplog.at_level(logging.WARNING):
        assert mod._swept_device_null(inp, sup, 5, 0.95, 0, 0) is None
    assert _warned(caplog, "swept device conditional perm-null failed")


def test_perm_null_gate_probe_failure_warns(monkeypatch, caplog):
    """A failing strict-resident probe disables the GPU perm-null and warns."""
    from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as mod
    from mlframe.feature_selection.filters import _gpu_strict_fe

    monkeypatch.delenv("MLFRAME_FE_CMI_PERM_NULL_GPU", raising=False)
    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_strict_resident_enabled", _boom)
    with caplog.at_level(logging.WARNING):
        assert mod.perm_null_gpu_resident_enabled() is False
    assert _warned(caplog, "fe_gpu_strict_resident_enabled() check failed")


def test_sparse_perm_null_failure_warns(monkeypatch, caplog):
    """A failing sparse perm-null returns None (the dense chunked path runs) and warns."""
    from mlframe.feature_selection.filters import _fe_cmi_perm_null_gpu as mod
    from mlframe.feature_selection.filters import _fe_resident_operands

    monkeypatch.setattr(_fe_resident_operands, "resident_operand", _boom)
    with caplog.at_level(logging.WARNING):
        out = mod._sparse_perm_null(np, np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), 4, 2, 1, None, None, None, None, 0.95)
    assert out is None
    assert _warned(caplog, "sparse perm-null CMI failed")


def test_gate_resident_probe_failure_warns(monkeypatch, caplog):
    """A failing resident-eligibility probe keeps the host path and warns."""
    from mlframe.feature_selection.filters import _fe_cmi_redundancy_gate as mod
    from mlframe.feature_selection.filters import _gpu_strict_fe

    monkeypatch.delenv("MLFRAME_FE_GATE_RESIDENT_CANDS", raising=False)
    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_strict_resident_enabled", _boom)
    with caplog.at_level(logging.WARNING):
        assert mod._cmi_gate_resident_eligible(np.zeros(4), ["a"]) is False
    assert _warned(caplog, "resident-GPU eligibility probe failed")


def test_gate_round_cmi_floor_failure_warns(monkeypatch, caplog):
    """A failing round CMI/floor computation yields empty tables and warns."""
    from mlframe.feature_selection.filters import _fe_batched_mi
    from mlframe.feature_selection.filters import _fe_cmi_redundancy_gate as mod

    _gpu_on(monkeypatch)
    monkeypatch.setattr(_fe_batched_mi, "batched_cmi_gpu", _boom)
    g = SimpleNamespace(y_dense=np.zeros(6, dtype=np.int64), n_rows=6, cand_bins={"a": np.zeros(6, dtype=np.int64), "b": np.ones(6, dtype=np.int64)}, cand_bins_dev=None)
    with caplog.at_level(logging.WARNING):
        out = mod._gate_round_batched(g, ["a", "b"], None, 0.95)
    assert out == ({}, {}, {})
    assert _warned(caplog, "the round CMI/floor computation failed")


def test_gate_device_binning_failure_warns(monkeypatch, caplog):
    """A failing device binning of a device-backed candidate warns and leaves no device code."""
    from mlframe.feature_selection.filters import _fe_cmi_redundancy_gate as mod

    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    with caplog.at_level(logging.WARNING):
        out = mod._apply_cmi_redundan_gate_resident_np_isfinite(True, SimpleNamespace(dev=object()), 4, "keep")
    assert out is None
    assert _warned(caplog, "device binning of a device-backed candidate failed")


def test_gate_gpu_resident_binning_failure_warns(monkeypatch, caplog):
    """A failing resident quantile binner warns and leaves no device code."""
    from mlframe.feature_selection.filters import _fe_cmi_redundancy_gate as mod
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe

    monkeypatch.setattr(_mi_greedy_cmi_fe, "_quantile_bin_gpu_resident", _boom)
    with caplog.at_level(logging.WARNING):
        out = mod._apply_cmi_redundan_gate_resident_np_isfinite(True, np.arange(8, dtype=np.float64), 4, "keep")
    assert out is None
    assert _warned(caplog, "_quantile_bin_gpu_resident failed")


def test_gate_device_join_failure_warns(monkeypatch, caplog):
    """A failing device join falls back to the host join and warns."""
    from mlframe.feature_selection.filters import _fe_cmi_redundancy_gate as mod
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe

    monkeypatch.setattr(_mi_greedy_cmi_fe, "_renumber_joint_gpu", _boom)
    a = np.array([0, 1, 0, 1], dtype=np.int64)
    b = np.array([0, 0, 1, 1], dtype=np.int64)
    with caplog.at_level(logging.WARNING):
        mod._joint_with_winner([a], [object()], b, object())
    assert _warned(caplog, "device join for the fragmentation check failed")


def test_gate_renumber_joint_gpu_failure_warns(monkeypatch, caplog):
    """A failing device renumber drops the device support and warns."""
    from mlframe.feature_selection.filters import _fe_cmi_redundancy_gate as mod
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe

    monkeypatch.setattr(_mi_greedy_cmi_fe, "_renumber_joint_gpu", _boom)
    with caplog.at_level(logging.WARNING):
        out = mod._apply_cmi_redundan_gate_resident_accepted_bins(True, [object()], "keep")
    assert out is None
    assert _warned(caplog, "_renumber_joint_gpu failed")


def test_qbin_fused_coder_failure_warns(monkeypatch, caplog):
    """A failing fused qbin coder falls back to the broadcast loop and warns."""
    from mlframe.feature_selection.filters import _fe_batched_qbin as mod

    monkeypatch.setattr(mod, "_get_qbin_coder_kernel", _boom)
    X = np.random.default_rng(0).normal(size=(32, 3))
    edges = np.quantile(X, np.linspace(0, 1, 5), axis=0)
    with caplog.at_level(logging.WARNING):
        codes = mod._codes_of_block(np, X, edges, False)
    assert codes.shape == (32, 3)
    assert _warned(caplog, "qbin fused coder failed")


def _stub_cupy(monkeypatch, count):
    """Install a stub cupy exposing only the device count."""
    from mlframe.feature_selection.filters import _gpu_policy

    cp = types.ModuleType("cupy")
    cp.cuda = SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: count))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(_gpu_policy, "gpu_globally_disabled", lambda: False)


def test_device_subset_parse_failure_warns(monkeypatch, caplog):
    """A malformed MLFRAME_FE_VRAM_DEVICES warns and keeps every visible device."""
    from mlframe.feature_selection.filters._fe_gpu_batch import _devices as mod

    _stub_cupy(monkeypatch, 2)
    monkeypatch.setenv("MLFRAME_FE_VRAM_DEVICES", "a,b")
    with caplog.at_level(logging.WARNING):
        assert mod._visible_device_ids() == [0, 1]
    assert _warned(caplog, "is not a comma-separated list of device ids")


def test_device_profile_failure_warns(monkeypatch, caplog):
    """A failing device profile skips that device and warns."""
    from mlframe.feature_selection.filters._fe_gpu_batch import _devices as mod

    monkeypatch.setattr(mod, "_visible_device_ids", lambda: [0])
    monkeypatch.setattr(mod, "_profile_device", _boom)
    with caplog.at_level(logging.WARNING):
        assert mod.enumerate_device_profiles() == []
    assert _warned(caplog, "profiling CUDA device 0 failed")
