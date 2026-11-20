"""Every host-or-device gate that falls back because its tuning lookup or GPU path failed says so at WARNING (driven on the CPU by monkeypatching)."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


class _Raising:
    """A tuner spec whose lookup always fails."""

    def choose(self, **kwargs):
        """Fail like a broken tuning cache."""
        raise OSError("cache unreadable")


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """Forget throttle counts so each warning is emitted."""
    reset_throttle_counts()


def _warned(caplog, needle: str) -> bool:
    """Whether a WARNING record mentions ``needle``."""
    return any(r.levelno == logging.WARNING and needle in r.getMessage() for r in caplog.records)


def test_residual_rescue_score_choose_failure_warns(monkeypatch, caplog):
    """A failing KTC lookup for the scoring backend falls to the floor with a WARNING."""
    from mlframe.feature_selection.filters import _residual_rescue_ktc as m

    monkeypatch.setattr(m, "_SPEC", _Raising())
    with caplog.at_level(logging.WARNING):
        assert m.score_device_chosen(10, 10) is False
    assert _warned(caplog, "residual_rescue_score_backend")


def test_residual_rescue_codes_choose_failure_warns(monkeypatch, caplog):
    """A failing KTC lookup for the coding backend falls to the floor with a WARNING."""
    from mlframe.feature_selection.filters import _residual_rescue_ktc as m

    monkeypatch.setattr(m, "_CODES_SPEC", _Raising())
    with caplog.at_level(logging.WARNING):
        assert m.codes_device_chosen(10, 10) is False
    assert _warned(caplog, "residual_rescue_codes_backend")


def test_rolling_window_choose_failure_warns(monkeypatch, caplog):
    """A failing KTC lookup for the rolling window falls to the row floor with a WARNING."""
    from mlframe.feature_selection.filters import _rolling_window_gpu as m

    monkeypatch.setattr(m, "_SPEC", _Raising())
    with caplog.at_level(logging.WARNING):
        assert m.rolling_device_chosen(10) is False
    assert _warned(caplog, "rolling_window_backend")


def test_row_argmax_choose_failure_warns(monkeypatch, caplog):
    """A failing KTC lookup for the row argmax falls to the row floor with a WARNING."""
    from mlframe.feature_selection.filters import _row_argmax_gpu as m

    monkeypatch.delenv("MLFRAME_ROW_ARGMAX_DEVICE", raising=False)
    monkeypatch.setattr(m, "_SPEC", _Raising())
    with caplog.at_level(logging.WARNING):
        assert m.row_argmax_on_device(10) is False
    assert _warned(caplog, "row_argmax_backend")


def test_scorer_device_selection_failure_warns(monkeypatch, caplog):
    """A failing kernel-tuning-cache load falls to the work-size fallback with a WARNING."""
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

    from mlframe.feature_selection.filters import _gpu_policy
    from mlframe.feature_selection.filters import _scorer_device_dispatch as m

    monkeypatch.setattr(_gpu_policy, "cuda_available_for_run", lambda: True)

    def _boom(*a, **k):
        """Fail like an unreadable cache."""
        raise OSError("cache unreadable")

    monkeypatch.setattr(KernelTuningCache, "load_or_create", staticmethod(_boom))
    with caplog.at_level(logging.WARNING):
        assert m.scorer_device_choice("probe", 10, 10, sweep=lambda: []) is False
    assert _warned(caplog, "backend selection failed")


def test_resident_candidate_strict_check_failure_warns(monkeypatch, caplog):
    """A failing STRICT-mode check continues to the crossover decision with a WARNING."""
    from mlframe.feature_selection.filters import _fe_gpu_strict, _gpu_policy
    from mlframe.feature_selection.filters import _resident_candidate_mi_ktc as m

    monkeypatch.setattr(_gpu_policy, "gpu_globally_disabled", lambda: False)

    def _boom(**kwargs):
        """Fail like a broken STRICT probe."""
        raise RuntimeError("strict probe broke")

    monkeypatch.setattr(_fe_gpu_strict, "fe_gpu_strict_enabled", _boom)
    monkeypatch.setattr(m, "_RESCAND_SPEC", None)
    with caplog.at_level(logging.WARNING):
        assert m.rescand_use_resident(1000, 16) is False
    assert _warned(caplog, "fe_gpu_strict_enabled() check failed")


def test_resident_raw_baseline_enabled_check_failure_warns(monkeypatch, caplog):
    """A failing enable check returns None (host fallback) with a WARNING."""
    from mlframe.feature_selection.filters import _gpu_strict_fe
    from mlframe.feature_selection.filters import _resident_raw_mi as m

    def _boom():
        """Fail like a broken enable probe."""
        raise RuntimeError("probe broke")

    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_resident_raw_baseline_enabled", _boom)
    with caplog.at_level(logging.WARNING):
        assert m.resident_raw_baseline_mi(np.ones((4, 2)), np.zeros(4), ("role", ("a", "b")), nbins=4) is None
    assert _warned(caplog, "fe_gpu_resident_raw_baseline_enabled() check failed")


def test_resident_raw_baseline_gpu_failure_warns(monkeypatch, caplog):
    """A failing resident upload returns None (host fallback) with a WARNING."""
    from mlframe.feature_selection.filters import _fe_resident_operands, _gpu_strict_fe
    from mlframe.feature_selection.filters import _resident_raw_mi as m

    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_resident_raw_baseline_enabled", lambda: True)
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))

    def _boom(*a, **k):
        """Fail like a device fault during the upload."""
        raise RuntimeError("device fault")

    monkeypatch.setattr(_fe_resident_operands, "assemble_resident_matrix", _boom)
    with caplog.at_level(logging.WARNING):
        assert m.resident_raw_baseline_mi(np.ones((4, 2)), np.zeros(4), ("role", ("a", "b")), nbins=4) is None
    assert _warned(caplog, "GPU path failed")
