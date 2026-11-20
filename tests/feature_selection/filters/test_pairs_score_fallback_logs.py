"""The pair-scoring fallback branches log a WARNING (driven on CPU by stubbing the failing call), so a silent slow path or a lost veto is visible."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts

_PKG = "mlframe.feature_selection.filters._feature_engineering_pairs"


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the per-key log counts so each test sees its first occurrence."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _warned(caplog, text):
    """Whether a WARNING record contains ``text``."""
    return any(r.levelno >= logging.WARNING and text in r.getMessage() for r in caplog.records)


def test_batched_abs_corr_device_fault_warns_and_returns_none(monkeypatch, caplog):
    """A device fault in the batched |corr| returns None for the per-column path and warns."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_steps as mod

    monkeypatch.setitem(sys.modules, "cupy", None)
    state = {"resolved_cols": {}}
    with caplog.at_level(logging.WARNING):
        out = mod._batched_abs_corr(state, np.zeros((4, 2)), [0, 1], np.zeros(4), np.ones(4, dtype=bool))
    assert out is None
    assert _warned(caplog, "batched device |corr| failed")


def test_noise_wrap_veto_failure_warns(caplog):
    """A fault while scoring the noise-wrap veto is logged and the pair is not vetoed."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_steps as mod

    st = SimpleNamespace(
        _passes_joint_gate=True, _prewarp_accept=False, _marginal_uplift_accept=False, _usability_accept=False, best_config=((0, 1), None, (0, 0, 0)), messages=[]
    )

    def _resolve(_cfg):
        """Fail like a broken column resolver."""
        raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING):
        mod._score_one_pair_step6_wide_fraction_margin(
            st, np.zeros(4), np.zeros((4, 1)), False, _resolve, lambda v: 0.0, {}, {}, {}, ("a", "b"), 0.5, 0.5, False, np.zeros((4, 1))
        )
    assert _warned(caplog, "noise-wrap corr-collapse veto failed")


def test_prefetch_submit_failure_warns(caplog):
    """A failed chunk prefetch submit is logged and the chunk is computed lazily."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_helpers as mod

    class _Ex:
        """An executor that refuses work."""

        def submit(self, *a, **k):
            """Fail like a shut-down pool."""
            raise RuntimeError("shutdown")

    with caplog.at_level(logging.WARNING):
        mod._score_one_pair_prefetch_next_chunk_into(_Ex(), [1, 2], 0, [0, 1], lambda *a: None, {})
    assert _warned(caplog, "chunk prefetch submit failed")


def test_producer_failure_warns(caplog):
    """A failed pipelined producer is logged and the cache is recomputed inline."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_helpers as mod

    class _Fut:
        """A future whose result raises."""

        def result(self):
            """Fail like a crashed producer."""
            raise RuntimeError("producer died")

    with caplog.at_level(logging.WARNING):
        assert mod._score_one_pair_fut_none(_Fut(), 3, {"x": 1}) is None
    assert _warned(caplog, "pipelined producer of chunk 3 failed")


def test_batched_pair_corr_failure_warns(caplog):
    """A failing batched |corr| of a pair is logged and scored per config."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_helpers as mod

    def _many(*a):
        """Fail like a device fault."""
        raise RuntimeError("batched broke")

    def _resolve(_c):
        """Resolve nothing, so the per-config loop skips every config."""
        return None

    _resolve.abs_corr_many = _many  # type: ignore[attr-defined]
    safe = lambda v: 0.0
    safe.target = (np.zeros(3), np.ones(3))  # type: ignore[attr-defined]
    with caplog.at_level(logging.WARNING):
        mod._score_one_pair_flips({(0, 0, 1): 1.0, (0, 0, 2): 1.0}, _resolve, safe, {})
    assert _warned(caplog, "batched |corr| of the pair failed")


def test_config_corr_failure_warns(caplog):
    """A failing per-config |corr| is logged and the config is left out of the map."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_helpers as mod

    def _resolve(_c):
        """Fail like a broken column resolver."""
        raise RuntimeError("cfg broke")

    use_map: dict = {}
    with caplog.at_level(logging.WARNING):
        mod._score_one_pair_flips({(0, 0, 1): 1.0}, _resolve, lambda v: 0.0, use_map)
    assert use_map == {}
    assert _warned(caplog, "|corr| of a pair config failed")


def test_resident_operand_table_build_failure_warns(monkeypatch, caplog):
    """A failing GPU-resident operand-table build is logged and the host H2D path is kept."""
    from mlframe.feature_selection.filters import _gpu_resident_fe, _gpu_resident_materialise
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_setup as mod

    def _fill(*args):
        """Record one operand so the resident build is attempted."""
        args[6][("v", "u")] = 0

    def _boom(*a, **k):
        """Fail like a device build error."""
        raise RuntimeError("build broke")

    monkeypatch.setattr(_gpu_resident_materialise, "fe_gpu_resident_operands_enabled", lambda: True)
    monkeypatch.setattr(_gpu_resident_fe, "_cuda_present", lambda: True)
    monkeypatch.setattr(_gpu_resident_fe, "build_resident_operand_table", _boom)
    monkeypatch.setattr(mod, "_materialise_pair_operands", _fill)
    monkeypatch.setattr(mod, "_operand_table_fits_vram", lambda tv: True)
    with caplog.at_level(logging.WARNING):
        out = mod._build_operand_table(
            prospective_pairs=[],
            transformed_vars=np.zeros((4, 2)),
            _unary_names_eff=[],
            unary_transformations={},
            _extval_raw_col=None,
            _prewarp_active=False,
            _prewarp_spec_by_var={},
            _gate_med_median_by_var={},
            cols=[],
            verbose=False,
            logger=logging.getLogger("t"),
            gpu_compatible_unary_names=[],
        )
    assert ("v", "u") in out
    assert _warned(caplog, "GPU-resident operand-table build failed")
