"""The device-failure fallbacks of the resident candidate-MI helpers and the KTC-choose fallbacks log at WARNING instead of debug."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_gpu_vram, _fe_resident_operands, _hermite_fe_mi
from mlframe.feature_selection.filters import _polynom_search_backend as pb
from mlframe.feature_selection.filters import _ratio_delta_screen_gpu as rsg
from mlframe.feature_selection.filters import _resident_candidate_mi as rcm
from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture
def fake_cupy(monkeypatch):
    """A bare ``cupy`` module: it imports, every device call on it raises, so the GPU branches fail the way a faulted device does."""
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _arrs():
    """Two small host operand columns and a binary target."""
    rng = np.random.default_rng(0)
    return {"a": rng.normal(size=64), "b": rng.normal(size=64)}, (rng.random(64) > 0.5).astype(np.int64)


def test_best_existing_op_mi_resident_logs_the_host_fallback(fake_cupy, caplog):
    """A failed device path returns None and says so at WARNING."""
    arrs, y = _arrs()
    with caplog.at_level(logging.WARNING, logger=rcm.logger.name):
        assert rcm.best_existing_op_mi_resident(arrs, ["a", "b"], y, 8) is None
    assert any("best_existing_op_mi_resident: GPU path failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_best_existing_op_mi_resident_many_logs_the_per_set_fallback(fake_cupy, caplog):
    """A failed batched device route returns None and says so at WARNING."""
    arrs, y = _arrs()
    with caplog.at_level(logging.WARNING, logger=rcm.logger.name):
        assert rcm.best_existing_op_mi_resident_many(arrs, [["a", "b"]], y, 8) is None
    assert any("device route failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_gate_grid_mi_resident_logs_the_host_fallback(fake_cupy, caplog):
    """A failed device tau-grid returns None and says so at WARNING."""
    arrs, y = _arrs()
    specs = [("mask", ("a", "b"), (arrs["b"], arrs["a"]), np.array([0.0, 0.5]))]
    with caplog.at_level(logging.WARNING, logger=rcm.logger.name):
        assert rcm.gate_grid_mi_resident(specs, y, 8) is None
    assert any("gate_grid_mi_resident: GPU path failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_best_existing_op_mi_resident_logs_a_missing_vram_cushion(fake_cupy, monkeypatch, caplog):
    """Declining the candidate block for lack of VRAM headroom is logged, not silent."""
    arrs, y = _arrs()
    monkeypatch.setattr(_fe_resident_operands, "resident_operand", lambda a, key, dtype=None: np.asarray(a))
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda nbytes: False)
    monkeypatch.setattr(_hermite_fe_mi, "relaxed_candidate_dtype", lambda cp: np.float32)
    monkeypatch.setattr(rcm, "crit_float_dtype", lambda: np.float32)
    with caplog.at_level(logging.WARNING, logger=rcm.logger.name):
        assert rcm.best_existing_op_mi_resident(arrs, ["a", "b"], y, 8) is None
    assert any("no VRAM cushion" in r.getMessage() for r in caplog.records)


def test_polynom_device_chosen_logs_a_failed_choose(monkeypatch, caplog):
    """A raising tuner falls back to the row floor and logs it at WARNING."""
    reset_throttle_counts()

    class _Boom:
        """Tuner whose lookup always fails."""

        def choose(self, **_kw):
            """Raise like a broken cache."""
            raise RuntimeError("cache broken")

    monkeypatch.setattr(pb, "_SPEC", _Boom())
    with caplog.at_level(logging.WARNING, logger=pb.logger.name):
        assert pb.polynom_device_chosen(pb.DEVICE_FALLBACK_ROWS) is True
    assert any("KTC choose() failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_ratio_pair_screen_on_device_logs_a_failed_choose(monkeypatch, caplog):
    """A raising tuner falls back to the row mark and logs it at WARNING."""
    reset_throttle_counts()

    class _Boom:
        """Tuner whose lookup always fails."""

        def choose(self, **_kw):
            """Raise like a broken cache."""
            raise RuntimeError("cache broken")

    monkeypatch.delenv("MLFRAME_RATIO_SCREEN_DEVICE", raising=False)
    monkeypatch.setattr(rsg, "_SPEC", _Boom())
    with caplog.at_level(logging.WARNING, logger=rsg.logger.name):
        assert rsg.ratio_pair_screen_on_device(rsg.DEVICE_FALLBACK_ROWS) is True
    assert any("KTC choose() failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
