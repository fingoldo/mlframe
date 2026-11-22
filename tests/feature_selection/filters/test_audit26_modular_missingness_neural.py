"""Regression tests: modular device-codes executor cleanup, missingness auto-detect log, MIST per-model calibration, DPMINE argument check."""
from __future__ import annotations

import concurrent.futures as cf
import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _missingness_fe as mf
from mlframe.feature_selection.filters import _modular_codes_gpu as mc
from mlframe.feature_selection.filters import _neural_mi as nm
from mlframe.utils.log_throttle import reset_throttle_counts


def _raise_launch_failure(*_a, **_k):
    """Raise like a failed kernel launch."""
    raise RuntimeError("launch failed")


def test_modular_prefetch_executor_is_shut_down_when_the_device_path_fails(monkeypatch):
    """A fault in the middle of the block loop must still shut the prefetch thread pool down."""
    shut = []

    class _Pool:
        """A pool spy that runs submitted work inline."""

        def __init__(self, max_workers=1):
            """Accept the real constructor's argument."""

        def submit(self, fn, *a):
            """Run the callable inline and wrap its result in a done future."""
            f: cf.Future = cf.Future()
            f.set_result(fn(*a))
            return f

        def shutdown(self, wait=True):
            """Record the shutdown."""
            shut.append(wait)

    fake_cp = types.ModuleType("cupy")
    fake_cp.asarray = lambda a: a  # type: ignore[attr-defined]
    fake_cp.empty = lambda *a, **k: None  # type: ignore[attr-defined]
    fake_cp.float32 = np.float32  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setattr(mc, "_use_device", lambda n: True)
    monkeypatch.setattr(mc, "kernel_for_source", lambda cp, real: _raise_launch_failure)
    monkeypatch.setattr(mc, "_sources_per_block", lambda *a, **k: 1)
    monkeypatch.setattr(mc, "block_size_for", lambda *a, **k: 32)
    monkeypatch.setattr(mc, "ThreadPoolExecutor", _Pool)
    X = pd.DataFrame({"a": np.arange(64, dtype=np.float32), "b": np.arange(64, dtype=np.float32)})
    assert mc.modular_family_codes_gpu(X, ["a", "b"], [7.0], 10) is None
    assert shut == [True]


class _FailingSpec:
    """A kernel-tuning spec whose choose() fails."""

    def choose(self, **_k):
        """Raise like a broken tuning cache."""
        raise RuntimeError("ktc broke")


def test_modular_choice_failure_is_logged(monkeypatch, caplog):
    """A failing tuning-cache choose() falls back to the element count and logs it at WARNING."""
    monkeypatch.setattr(mc, "_SPEC", _FailingSpec())
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=mc.logger.name):
        assert mc.modular_device_chosen(10) is False
    assert any("choose() failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_auto_detect_missing_cols_logs_a_column_it_cannot_inspect(caplog):
    """A column whose isna() raises is skipped with a WARNING instead of a debug line."""

    class _Bad:
        """A column whose isna() fails."""

        def isna(self):
            """Raise like a broken column."""
            raise TypeError("cannot inspect")

    class _Frame(pd.DataFrame):
        """A frame whose column 'bad' is unreadable."""

        def __getitem__(self, key):
            """Return the broken column for 'bad'."""
            return _Bad() if key == "bad" else super().__getitem__(key)

    X = _Frame({"ok": [1.0, np.nan, 3.0, np.nan], "bad": [1, 2, 3, 4]})
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=mf.logger.name):
        assert mf.auto_detect_missing_cols(X, min_nan_rate=0.1) == ["ok"]
    assert any("isna()" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_mist_qr_calibration_is_built_from_the_qr_model(monkeypatch):
    """Calibrating the quantile-regression model must use that model's raw output, not the MSE model's."""
    asked: list[str] = []

    class _Model:
        """A fake MIST model whose raw output is the sample correlation."""

        def __init__(self, loss):
            """Remember which head this is."""
            self.loss = loss

        def estimate_point(self, X, Y):
            """Raw score: correlation, scaled differently per head."""
            r = float(np.corrcoef(np.asarray(X).ravel(), np.asarray(Y).ravel())[0, 1])
            return r * (1.0 if self.loss == "mse" else 5.0)

    def fake_get(loss="mse", device="auto"):
        """Record the requested head."""
        asked.append(loss)
        return _Model(loss)

    monkeypatch.setattr(nm, "_get_mist_hf_model", fake_get)
    monkeypatch.setattr(nm, "_MIST_CALIBRATION_CACHE", {})
    raw_qr, _ = nm._calibrate_mist(device="cpu", N=300, n_calibration_per_rho=1, y_kind="continuous", loss="qr")
    raw_mse, _ = nm._calibrate_mist(device="cpu", N=300, n_calibration_per_rho=1, y_kind="continuous", loss="mse")
    assert asked == ["qr", "mse"]
    np.testing.assert_allclose(raw_qr, 5.0 * raw_mse, rtol=1e-12, err_msg="the QR calibration is not the QR model's raw output")


def test_mist_mi_passes_its_loss_to_the_calibration(monkeypatch):
    """``mist_mi(loss='qr', calibrated=True)`` requests the QR calibration table."""
    seen = {}

    class _Model:
        """A constant fake model."""

        def estimate_point(self, X, Y):
            """A fixed raw score."""
            return 0.3

    monkeypatch.setattr(nm, "_get_mist_hf_model", lambda loss="mse", device="auto": _Model())

    def fake_cal(device="auto", N=2000, seed=42, n_calibration_per_rho=3, y_kind="continuous", loss="mse"):
        """Record the loss the table is requested for."""
        seen["loss"] = loss
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    monkeypatch.setattr(nm, "_calibrate_mist", fake_cal)
    rng = np.random.default_rng(0)
    nm.mist_mi(rng.standard_normal(100), rng.standard_normal(100), loss="qr", calibrated=True, device="cpu")
    assert seen["loss"] == "qr"


def test_dpmine_rejects_zero_iterations():
    """``n_iter=0`` used to die with an UnboundLocalError after training nothing."""
    pytest.importorskip("torch")
    x = np.random.default_rng(0).standard_normal(64)
    with pytest.raises(ValueError, match="n_iter"):
        nm.dpmine_mi(x, x, n_iter=0, device="cpu")
