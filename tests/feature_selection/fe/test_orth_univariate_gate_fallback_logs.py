"""Gate and device fallbacks of the orthogonal-univariate FE modules announce themselves (WARNING), driven on the CPU by monkeypatching."""

import logging
import types

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_gpu_resident as gpu_res
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _gpu_resident_cross_basis as xbasis
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_abs_floor
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_dedup
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe as eb


def _raiser(exc):
    """A callable that raises ``exc`` whatever it is called with."""

    def _f(*_a, **_k):
        """Raise the captured exception."""
        raise exc

    return _f


def test_resolve_detect_dtype_honours_criterion_dtype(monkeypatch):
    """The detector operands use the criterion dtype, not an unconditional float64."""
    monkeypatch.setattr(gpu_res, "crit_float_dtype", lambda: np.float32)
    assert gpu_res._resolve_detect_dtype(types.SimpleNamespace(float64=np.float64)) is np.float32


def test_resolve_detect_dtype_lookup_failure_logs_and_uses_float64(monkeypatch, caplog):
    """A failing criterion-dtype lookup falls back to float64 and says so."""
    monkeypatch.setattr(gpu_res, "crit_float_dtype", _raiser(RuntimeError("boom")))
    with caplog.at_level(logging.WARNING):
        got = gpu_res._resolve_detect_dtype(types.SimpleNamespace(float64=np.float64))
    assert got is np.float64
    assert "crit_float_dtype() resolution failed" in caplog.text


def test_deflate_sincos_gpu_singular_design_logs(monkeypatch, caplog):
    """An unusable [1, sin, cos] design returns y undeflated and warns."""
    monkeypatch.setenv("MLFRAME_FE_FOURIER_FUSED", "0")
    err = np.linalg.LinAlgError("singular")
    cp = types.SimpleNamespace(
        ones_like=np.ones_like, sin=np.sin, cos=np.cos, stack=np.stack, linalg=types.SimpleNamespace(solve=_raiser(err), lstsq=_raiser(err)),
    )
    z = np.linspace(0.0, 1.0, 50)
    y = np.arange(50.0)
    with caplog.at_level(logging.WARNING):
        out = gpu_res._deflate_sincos_gpu(cp, z, y, 1.0)
    assert out is y
    assert "_deflate_sincos_gpu fell back" in caplog.text


def test_try_gpu_fourier_detect_device_fault_logs_and_returns_none(monkeypatch, caplog):
    """A genuine device/linalg fault in the resident detector is logged before the CPU detector takes over."""
    import mlframe.feature_selection.filters._gpu_strict_fe._entry as entry

    monkeypatch.setattr(entry, "fe_gpu_strict_resident_enabled", lambda: True)
    monkeypatch.setattr(gpu_res, "detect_fourier_freqs_for_col_gpu", _raiser(np.linalg.LinAlgError("dev")))
    z = np.linspace(0.0, 1.0, 900)
    with caplog.at_level(logging.WARNING):
        got = eb._try_gpu_fourier_detect(z, np.sin(z), (1.0, 2.0), 0.15, 800, 2)
    assert got is None
    assert "GPU-resident Fourier detector failed" in caplog.text


def test_polynomial_detrend_failure_logs_and_keeps_target(monkeypatch, caplog):
    """A failing cubic fit leaves y undetrended and warns instead of logging at debug."""
    err = np.linalg.LinAlgError("no convergence")
    monkeypatch.setattr(np.linalg, "solve", _raiser(err))
    monkeypatch.setattr(np.linalg, "lstsq", _raiser(err))
    z = np.linspace(0.0, 1.0, 40)
    y = np.arange(40.0)
    with caplog.at_level(logging.WARNING):
        y_tr, _y_va = eb._polynomial_detrend(z, z[:10], y, y[:10])
    assert y_tr is y
    assert "_polynomial_detrend: cubic fit failed" in caplog.text


def test_dedup_strict_probe_failure_logs(monkeypatch, caplog):
    """A failing strict-mode probe keeps the default dedup backend and warns."""
    import mlframe.feature_selection.filters._fe_gpu_strict as strict

    monkeypatch.delenv("MLFRAME_FE_DEDUP_CORR_BACKEND", raising=False)
    monkeypatch.setattr(strict, "fe_gpu_strict_enabled", _raiser(RuntimeError("probe")))
    monkeypatch.setattr(_orth_dedup, "_SPEC", None)
    with caplog.at_level(logging.WARNING):
        assert _orth_dedup._resolve_pc_backend(10, 10, 100) == "numpy"
    assert "fe_gpu_strict_enabled() probe failed" in caplog.text


def test_crossbasis_ktc_choice_failure_logs(monkeypatch, caplog):
    """A failing kernel-tuning lookup falls to the row floor and warns."""
    monkeypatch.setattr(xbasis, "_SPEC", types.SimpleNamespace(choose=_raiser(RuntimeError("ktc"))))
    with caplog.at_level(logging.WARNING):
        assert xbasis.crossbasis_device_chosen(10) is False
    assert "KTC choose() failed" in caplog.text


def test_crossbasis_resident_scoring_failure_logs_and_returns_none(monkeypatch, caplog):
    """A failing device-born scoring returns None for the host scorer and warns."""
    monkeypatch.setitem(__import__("sys").modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(xbasis, "crossbasis_device_chosen", lambda n: True)
    monkeypatch.setattr(xbasis, "_product_mi_blocked", _raiser(RuntimeError("oom")))
    raw = pd.DataFrame({"a": np.arange(10.0)})
    eng = pd.DataFrame({"a__x": np.arange(10.0)})
    with caplog.at_level(logging.WARNING):
        got = xbasis.raw_and_product_mi_resident(raw, eng, np.zeros(10, dtype=int), [{"legs": [("a", 1)]}], nbins=4)
    assert got is None
    assert "GPU path failed" in caplog.text


def test_abs_floor_drop_of_unsatisfiable_mad_floor_logs(caplog):
    """Dropping a MAD floor that sits at or above H(y) is announced."""
    scores = pd.DataFrame({"baseline_mi": [0.1, 0.9, 0.1, 0.9, 0.5], "engineered_mi": [0.1, 0.9, 0.1, 0.9, 0.5]})
    with caplog.at_level(logging.WARNING):
        floor = _orth_abs_floor.hybrid_abs_mi_floor(scores, 0.1, 0.69)
    assert abs(floor - 0.09) < 1e-12
    assert "dropped a MAD noise floor" in caplog.text
