"""Live-knob pins for PreprocessingBackendConfig.fallback_to_sklearn.

``fallback_to_sklearn`` is now consumed in ``fit_and_transform_pipeline``: when a
polars input is routed to the polars-ds backend but polars-ds is unavailable (the
pipeline builds to ``None``), an enabled flag converts the splits to pandas and
runs the sklearn pandas backend instead of silently passing the frame through raw.

These tests pin the LIVE behaviour: the field is read at the dispatch site, and
the True/False values diverge on a polars input when polars-ds is unavailable.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import polars as pl

import mlframe
from mlframe.training.configs import PreprocessingBackendConfig
from mlframe.training.pipeline import _pipeline_fit_transform


def _src_root() -> pathlib.Path:
    return pathlib.Path(mlframe.__file__).resolve().parent


def test_fallback_to_sklearn_is_consumed_at_dispatch_site():
    """Sensor: the dispatch module reads ``config.fallback_to_sklearn``. Zero
    consumers would mean the knob regressed back to dead -- fail loudly if so.
    """
    text = pathlib.Path(_pipeline_fit_transform.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "config.fallback_to_sklearn" in text, "fallback_to_sklearn is no longer read at the dispatch site -- the knob regressed to dead."


def _force_no_polarsds(monkeypatch):
    """Make ``create_polarsds_pipeline`` behave as if polars-ds is unavailable."""
    monkeypatch.setattr(
        "mlframe.training.pipeline.create_polarsds_pipeline",
        lambda *a, **k: None,
    )


def _toy_polars_frame() -> pl.DataFrame:
    return pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})


def test_fallback_engages_when_true_and_backend_unavailable(monkeypatch):
    _force_no_polarsds(monkeypatch)
    cfg = PreprocessingBackendConfig(prefer_polarsds=True, fallback_to_sklearn=True, scaler_name=None, imputer_strategy=None, categorical_encoding=None)
    train, _, _, _, _ = _pipeline_fit_transform.fit_and_transform_pipeline(
        _toy_polars_frame(),
        None,
        None,
        cfg,
        ensure_float32=False,
        verbose=0,
    )
    # Fallback converted the polars input to a pandas frame via the sklearn backend.
    assert isinstance(train, pd.DataFrame), "fallback_to_sklearn=True should route the polars input through the pandas backend"


def test_no_fallback_when_false_keeps_polars(monkeypatch):
    _force_no_polarsds(monkeypatch)
    cfg = PreprocessingBackendConfig(prefer_polarsds=True, fallback_to_sklearn=False, scaler_name=None, imputer_strategy=None, categorical_encoding=None)
    train, _, _, _, _ = _pipeline_fit_transform.fit_and_transform_pipeline(
        _toy_polars_frame(),
        None,
        None,
        cfg,
        ensure_float32=False,
        verbose=0,
    )
    # No fallback: the frame stays polars (current pre-wiring behaviour preserved).
    assert isinstance(train, pl.DataFrame), "fallback_to_sklearn=False must preserve the raw polars pass-through (no sklearn fallback)"
