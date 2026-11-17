"""GPU-resident FE helpers: slow-path fallbacks say so at WARNING, and out-of-range inputs to compiled kernels are refused up front.

Failures are injected by monkeypatching, so each test drives the changed branch itself; tests that need a device skip without one.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttle_counts():
    """Forget throttle counts so each test sees its first warning."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


@pytest.fixture
def cp():
    """The cupy module, skipping the test when there is no usable device."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("no CUDA device")
    return cupy


def _warnings(caplog):
    """The WARNING-level messages captured so far."""
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_column_major_materialise_failure_warns_and_uses_row_major(cp, monkeypatch, caplog):
    """A failing column-major build warns and the row-major kernel still returns the right matrix."""
    from mlframe.feature_selection.filters import _gpu_resident_materialise as m

    def _boom(*_a, **_k):
        """Fail like a transpose/launch fault."""
        raise RuntimeError("cm transpose fault")

    monkeypatch.setattr(m, "_operand_table_cm", _boom)
    tv = cp.asarray(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
    with caplog.at_level(logging.WARNING, logger=m.logger.name):
        out = m._fe_materialise_block_gpu(tv, [0], [1], np.array([0], dtype=np.int8))
    assert cp.asnumpy(out).tolist() == [[6.0], [20.0]]
    assert any("column-major fe_materialise failed" in x for x in _warnings(caplog))


def test_pinned_d2h_failure_warns_and_uses_pageable_copy(cp, monkeypatch, caplog):
    """A failing pinned staging buffer warns and the candidate matrix still arrives through cp.asnumpy."""
    from mlframe.feature_selection.filters import _gpu_resident_materialise as m

    def _boom(*_a, **_k):
        """Fail like a pinned allocation fault."""
        raise RuntimeError("pinned alloc fault")

    monkeypatch.setattr(m, "_pinned_view", _boom)
    tv = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]], dtype=np.float32)
    out_cand = np.zeros((4, 1), dtype=np.float32)
    with caplog.at_level(logging.WARNING, logger=m.logger.name):
        m.gpu_materialise_discretize_codes_host(tv, np.array([0]), np.array([1]), np.array([0], dtype=np.int8), 2, out_cand=out_cand)
    assert out_cand[:, 0].tolist() == [6.0, 20.0, 42.0, 72.0]
    assert any("pinned D2H staging failed" in x for x in _warnings(caplog))


def test_operand_backend_choose_failure_warns(monkeypatch, caplog):
    """A failing tuner lookup for the operand-table backend uses the row floor and says why."""
    from mlframe.feature_selection.filters import _gpu_resident_operand_build as ob

    class _Raising:
        """A tuning spec whose lookup fails."""

        def choose(self, **_k):
            """Raise like an unreadable cache."""
            raise RuntimeError("tuning cache unreadable")

    monkeypatch.setattr(ob, "_SPEC", _Raising())
    with caplog.at_level(logging.WARNING, logger=ob.logger.name):
        assert ob.operand_table_device_chosen(ob.DEVICE_FALLBACK_ROWS) is True
    assert any("KTC choose() failed" in x and "tuning cache unreadable" in x for x in _warnings(caplog))


def test_gpu_pairs_fe_mi_failure_warns_and_returns_none(cp, monkeypatch, caplog):
    """A fault inside the analytic GPU pair-MI path warns and hands the caller None (the CPU dispatcher)."""
    from mlframe.feature_selection.filters import _analytic_mi_null as an
    from mlframe.feature_selection.filters import _fe_gpu_vram
    from mlframe.feature_selection.filters import _gpu_resident_discretize as gd
    from mlframe.feature_selection.filters import _gpu_resident_pair_mi as pm

    monkeypatch.setattr(an, "analytic_null_enabled", lambda: True)
    monkeypatch.setattr(an, "analytic_null_applicable", lambda *a, **k: True)
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *_a, **_k: True)

    def _boom(*_a, **_k):
        """Fail like a kernel fault."""
        raise RuntimeError("binning kernel fault")

    monkeypatch.setattr(gd, "_gpu_resident_discretize_codes", _boom)
    yc = np.arange(100) % 2
    with caplog.at_level(logging.WARNING, logger=pm.__name__):
        res = pm.gpu_pairs_fe_mi(np.random.default_rng(0).random((100, 3)).astype(np.float32), 5, yc, yc, np.array([0.5, 0.5]), 3, 0.0, False)
    assert res is None
    assert any("gpu_pairs_fe_mi failed" in x for x in _warnings(caplog))


def test_gpu_pairs_fe_mi_vram_refusal_warns(monkeypatch, caplog):
    """A refusal by the VRAM cushion check warns before returning None."""
    from mlframe.feature_selection.filters import _analytic_mi_null as an
    from mlframe.feature_selection.filters import _fe_gpu_vram
    from mlframe.feature_selection.filters import _gpu_resident_pair_mi as pm

    monkeypatch.setattr(an, "analytic_null_enabled", lambda: True)
    monkeypatch.setattr(an, "analytic_null_applicable", lambda *a, **k: True)
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *_a, **_k: False)
    pytest.importorskip("cupy")
    yc = np.arange(100) % 2
    with caplog.at_level(logging.WARNING, logger=pm.__name__):
        res = pm.gpu_pairs_fe_mi(np.zeros((100, 3), dtype=np.float32), 5, yc, yc, np.array([0.5, 0.5]), 3, 0.0, False)
    assert res is None
    assert any("VRAM cushion" in x for x in _warnings(caplog))


@pytest.mark.parametrize("fn_name, text", [("fe_gpu_pairs_mi_backend_choice", "fe_gpu_pairs_mi_backend_choice"), ("fe_gpu_binning_backend_choice", "fe_gpu_binning_backend_choice")])
def test_backend_choice_ktc_failure_warns(monkeypatch, caplog, fn_name, text):
    """A failing kernel-tuning-cache lookup uses the size fallback and says so at WARNING."""
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

    from mlframe.feature_selection.filters import _fe_deadline
    from mlframe.feature_selection.filters import _gpu_resident_pair_mi as pm

    def _boom(*_a, **_k):
        """Fail like an unreadable cache file."""
        raise RuntimeError("cache unreadable")

    monkeypatch.setattr(_fe_deadline, "fe_budget_active", lambda: False)
    monkeypatch.setattr(KernelTuningCache, "load_or_create", classmethod(_boom))
    with caplog.at_level(logging.WARNING, logger=pm.__name__):
        assert getattr(pm, fn_name)(10, 10) in ("cpu", "gpu")
    assert any(text in x and "lookup failed" in x for x in _warnings(caplog))


def test_grand_fusion_fused_failure_warns_before_non_fused_path(monkeypatch, caplog):
    """When the fused grand-fusion path raises, the slower non-fused path is announced at WARNING."""
    from mlframe.feature_selection.filters import _gpu_resident_pair_mi as pm

    def _boom(*_a, **_k):
        """Fail like a shared-memory overflow."""
        raise RuntimeError("shared histogram too large")

    monkeypatch.setattr(pm, "fe_gpu_grand_fusion_enabled", lambda: True)
    monkeypatch.setattr(pm, "grand_fused_pair_mi_fused", _boom)
    monkeypatch.setattr(pm, "_grand_fused_pair_mi_part1", lambda *a, **k: (True, (["n"], np.zeros(1))))
    with caplog.at_level(logging.WARNING, logger=pm.__name__):
        names, _mi = pm.grand_fused_pair_mi(np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(2))
    assert names == ["n"]
    assert any("fused path unavailable" in x and "shared histogram too large" in x for x in _warnings(caplog))


def test_non_fused_grand_fusion_rejects_nbins_beyond_int8_codes():
    """nbins above 128 cannot be held by the int8 resident codes and is refused instead of wrapping negative."""
    from mlframe.feature_selection.filters import _gpu_resident_pair_mi_parts as parts

    with pytest.raises(ValueError, match="int8"):
        parts._grand_fused_pair_mi_part1(np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(2), 200, False, 3, 0, 0.0)


def test_rank_mi_gpu_failure_warns_and_returns_none(cp, monkeypatch, caplog):
    """A fault in the rank-binned GPU MI warns (it falls back to the host rank path) and returns None."""
    from mlframe.feature_selection.filters import _gpu_resident_rank_bin as rb

    def _boom(*_a, **_k):
        """Fail like a sort fault."""
        raise RuntimeError("argsort fault")

    monkeypatch.setattr(rb, "rank_bin_codes_batch_gpu_resident", _boom)
    X = cp.asarray(np.random.default_rng(0).random((50, 2)))
    y = cp.asarray(np.arange(50) % 2)
    with caplog.at_level(logging.WARNING, logger=rb.logger.name):
        assert rb.plugin_mi_classif_batch_rank_cuda_resident(X, y, 4) is None
    assert any("argsort fault" in x and "host fallback" in x for x in _warnings(caplog))


def test_radix_quantiles_refuse_fractions_outside_unit_interval(cp):
    """A quantile fraction outside [0, 1] would address a rank beyond the column and is refused."""
    from mlframe.feature_selection.filters import _gpu_resident_select as sel

    with pytest.raises(ValueError, match="fractions"):
        sel._radix_quantiles(cp.zeros((10, 2), dtype=cp.float64), [0.5, 1.5])


def test_radix_f64_v3_scratch_oom_warns_and_uses_v2(cp, monkeypatch, caplog):
    """A real out-of-memory in the v3 compaction launch is logged (only the budget refusal used to be) and the v2 kernel finishes."""
    from mlframe.feature_selection.filters import _gpu_resident_select as sel

    def _oom(*_a, **_k):
        """Fail like an allocation that does not fit."""
        raise MemoryError("out of memory")

    monkeypatch.setattr(sel, "_get_radix_select_interp_f64_v3_kernel", lambda: _oom)
    monkeypatch.setattr(sel, "_scratch_fits", lambda _b: True)
    cand = cp.asarray(np.random.default_rng(1).random((200, 3)))
    with caplog.at_level(logging.WARNING, logger=sel.logger.name):
        edges = sel._radix_select_interior_edges(cand, 4)
    assert edges is not None and edges.shape == (3, 3)
    assert any("ran out of memory" in x for x in _warnings(caplog))


def test_v6_first_digit_bits_rejects_malformed_values(caplog):
    """A malformed or out-of-range MLFRAME_RADIX_F32_V6_BITS falls back to 13 with a warning instead of breaking the import."""
    from mlframe.feature_selection.filters import _gpu_resident_select_f32_wide as wide

    with caplog.at_level(logging.WARNING, logger=wide.__name__):
        assert wide._first_digit_bits("abc") == 13
        assert wide._first_digit_bits("20") == 13
    assert wide._first_digit_bits("11") == 11
    assert sum("MLFRAME_RADIX_F32_V6_BITS" in x for x in _warnings(caplog)) >= 1
