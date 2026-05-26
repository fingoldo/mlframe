"""Regression sensors for Wave 16B: bruteforce.run_pysr_feature_engineering polars-side fillna, byte-size input gate, and caller-frame immutability through the rename + cat-cast path. Covers A2#6 and A2#11.

Mocks ``pysr.PySRRegressor`` so Julia never starts -- these are pure-Python sensors that exercise the pre-PySR pandas / polars plumbing.
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest


# Install a fake ``pysr`` module BEFORE importing the bruteforce module so the lazy ``from pysr import PySRRegressor`` inside the function picks up our stub instead of starting Julia.
@pytest.fixture(autouse=True)
def _stub_pysr(monkeypatch):
    fake_model = MagicMock()
    fake_model.fit = MagicMock(return_value=None)
    fake_model.get_best = MagicMock(return_value="x0")
    # ``equations.equation.tolist()`` chain used by the verbose-logging branch.
    fake_eqs = MagicMock()
    fake_eqs.equation.tolist = MagicMock(return_value=["x0"])
    fake_model.equations = fake_eqs
    # Track which columns PySR's ``fit`` actually saw.
    fake_model.feature_names_in_ = None

    def _make_regressor(**_kwargs):
        # On every fit, capture the column order from the X passed.
        def _fake_fit(X, y):
            if hasattr(X, "columns"):
                fake_model.feature_names_in_ = np.array(list(X.columns))
            else:
                fake_model.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])
        fake_model.fit.side_effect = _fake_fit
        return fake_model

    fake_pysr = types.ModuleType("pysr")
    fake_pysr.PySRRegressor = _make_regressor
    monkeypatch.setitem(sys.modules, "pysr", fake_pysr)
    yield fake_model


def _import_target():
    from mlframe.feature_engineering import bruteforce as bf
    return bf


# ---------- A2#6 sensors ----------

def test_pysr_input_refused_above_byte_limit(_stub_pysr, monkeypatch):
    """Byte-size gate trips when the sampled frame exceeds ``MLFRAME_PYSR_INPUT_BYTES_LIMIT``."""
    bf = _import_target()
    monkeypatch.setenv("MLFRAME_PYSR_INPUT_BYTES_LIMIT", "1024")  # 1 KB -- guaranteed under-sized
    df = pd.DataFrame({
        "f0": np.random.default_rng(0).random(2000),
        "f1": np.random.default_rng(1).random(2000),
        "target": np.random.default_rng(2).random(2000),
    })
    with pytest.raises(ValueError, match=r"exceeds the 1,024-byte limit"):
        bf.run_pysr_feature_engineering(df, target_col="target", sample_size=2000, verbose=0)


def test_pysr_input_refused_above_byte_limit_polars(_stub_pysr, monkeypatch):
    """Same byte-size gate trips on the polars-branch input."""
    bf = _import_target()
    monkeypatch.setenv("MLFRAME_PYSR_INPUT_BYTES_LIMIT", "1024")
    df = pl.DataFrame({
        "f0": np.random.default_rng(0).random(2000),
        "f1": np.random.default_rng(1).random(2000),
        "target": np.random.default_rng(2).random(2000),
    })
    with pytest.raises(ValueError, match=r"exceeds the 1,024-byte limit"):
        bf.run_pysr_feature_engineering(df, target_col="target", sample_size=2000, verbose=0)


def test_pysr_input_byte_limit_env_default_is_2gb(_stub_pysr):
    """Default limit is the documented 2 GB so casual callers don't trip it."""
    bf = _import_target()
    assert bf._DEFAULT_PYSR_INPUT_BYTES_LIMIT == 2_000_000_000


def test_bruteforce_polars_side_fillna_no_pandas_broadcast(_stub_pysr, monkeypatch):
    """Polars-branch input: ``pd.DataFrame.fillna`` must NOT be called inside ``run_pysr_feature_engineering`` -- fill happens on the polars side before ``to_pandas()``."""
    bf = _import_target()

    original_fillna = pd.DataFrame.fillna
    calls = {"count": 0, "stacks": []}

    def _spy_fillna(self, *args, **kwargs):
        import traceback
        stack = traceback.extract_stack()
        # only count calls fired from inside bruteforce.run_pysr_feature_engineering
        if any("bruteforce.py" in frame.filename and "run_pysr_feature_engineering" in frame.name for frame in stack):
            calls["count"] += 1
            calls["stacks"].append([(f.filename, f.name, f.lineno) for f in stack[-5:]])
        return original_fillna(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "fillna", _spy_fillna)

    rng = np.random.default_rng(0)
    arr = rng.random(50)
    arr[5:10] = np.nan
    df = pl.DataFrame({
        "f0": arr,
        "f1": rng.random(50),
        "target": rng.random(50),
    })

    bf.run_pysr_feature_engineering(df, target_col="target", sample_size=50, verbose=0)

    assert calls["count"] == 0, (
        f"A2#6: pd.DataFrame.fillna called {calls['count']} times inside run_pysr_feature_engineering "
        f"on polars input; the polars-side imputation should make this unnecessary. Stacks: {calls['stacks']}"
    )


def test_bruteforce_polars_side_fillna_imputes_with_median(_stub_pysr, monkeypatch):
    """Sanity: the polars-side fill replaces NaN with the per-column median (not 0). Recovered by inspecting the X handed to the stubbed fit."""
    bf = _import_target()

    captured = {}

    def _make_regressor(**_kwargs):
        fake = MagicMock()
        def _fake_fit(X, y):
            captured["X"] = X.copy() if hasattr(X, "copy") else X
        fake.fit.side_effect = _fake_fit
        fake.get_best.return_value = "x0"
        eqs = MagicMock()
        eqs.equation.tolist.return_value = ["x0"]
        fake.equations = eqs
        return fake

    monkeypatch.setitem(sys.modules, "pysr", types.SimpleNamespace(PySRRegressor=_make_regressor))

    arr = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0])  # median over the 9 non-NaN values is 5.5
    df = pl.DataFrame({
        "f0": arr,
        "f1": np.arange(10, dtype=np.float64),
        "target": np.arange(10, dtype=np.float64) * 2.0,
    })

    bf.run_pysr_feature_engineering(df, target_col="target", sample_size=10, random_state=0, verbose=0)

    X = captured["X"]
    assert "f0" in X.columns
    f0 = X["f0"].to_numpy()
    # Index 4 was NaN; should now be the median of the remaining 9 values.
    expected_median = float(np.nanmedian(arr))
    assert np.isfinite(f0).all(), "NaN should be filled, not zeroed-out or left as NaN"
    assert abs(float(f0[4]) - expected_median) < 1e-9, (
        f"A2#6 imputation: f0[4] should be median ({expected_median}), got {f0[4]}"
    )


# ---------- A2#11 sensors ----------

def test_run_pysr_fe_does_not_rename_caller_frame(_stub_pysr):
    """Caller's pandas DataFrame (passed as a head-view) must keep its original column names after PySR fit. Pre-fix the ``tmp_df.columns = _final_names`` mutation aliased the caller's column Index when ``df.sample()`` returned a frame that shared the column-Index object."""
    bf = _import_target()
    original = pd.DataFrame({
        "feat-A": np.arange(100, dtype=np.float64),  # "-" gets sanitised to "_"
        "feat=B": np.arange(100, dtype=np.float64),  # "=" gets sanitised to "_"
        "target": np.arange(100, dtype=np.float64),
    })
    # Caller passes a view-like slice (df.head returns a view in pandas).
    view = original.head(30)
    cols_before = list(original.columns)
    view_cols_before = list(view.columns)

    bf.run_pysr_feature_engineering(view, target_col="target", sample_size=30, verbose=0)

    assert list(original.columns) == cols_before, (
        f"A2#11: caller's original.columns changed: before={cols_before} after={list(original.columns)}"
    )
    assert list(view.columns) == view_cols_before, (
        f"A2#11: caller's view.columns changed: before={view_cols_before} after={list(view.columns)}"
    )


def test_run_pysr_fe_model_sees_sanitised_names(_stub_pysr):
    """The PySR model's fit() must receive a frame whose columns have ``-`` and ``=`` replaced with ``_``. Pre-fix this happens in-place on the caller's columns; post-fix it happens on the owned tmp_df."""
    bf = _import_target()
    df = pd.DataFrame({
        "alpha-x": np.arange(50, dtype=np.float64),
        "beta=y": np.arange(50, dtype=np.float64),
        "target": np.arange(50, dtype=np.float64),
    })

    fe_model = bf.run_pysr_feature_engineering(df, target_col="target", sample_size=50, verbose=0)

    seen = list(fe_model.feature_names_in_)
    assert "alpha_x" in seen, f"PySR should see sanitised name 'alpha_x'; got {seen}"
    assert "beta_y" in seen, f"PySR should see sanitised name 'beta_y'; got {seen}"
    assert "alpha-x" not in seen
    assert "beta=y" not in seen


def test_run_pysr_fe_no_view_aliasing_with_polars(_stub_pysr):
    """Polars-branch: caller's polars frame columns must be unchanged after fit (polars rename is on the post-``to_pandas()`` tmp_df, so polars side is naturally safe -- this test pins it)."""
    bf = _import_target()
    original = pl.DataFrame({
        "feat-A": np.arange(100, dtype=np.float64),
        "feat=B": np.arange(100, dtype=np.float64),
        "target": np.arange(100, dtype=np.float64),
    })
    cols_before = list(original.columns)

    bf.run_pysr_feature_engineering(original, target_col="target", sample_size=30, verbose=0)

    assert list(original.columns) == cols_before, (
        f"A2#11 (polars branch): caller's polars columns changed: before={cols_before} after={list(original.columns)}"
    )


# ---------- PySR-polars-finding sensor ----------

def test_pysr_polars_native_support_documented():
    """The Wave 16B manifest must document the PySR polars-input investigation finding so future readers know why we keep the ``to_pandas()`` boundary.

    PySR 1.5.5 docstring says ``X : ndarray | pandas.DataFrame`` and the source does ``isinstance(X, pd.DataFrame)`` -- the polars-only space-rename safety branch is SKIPPED on polars input. sklearn validate_data DOES accept polars in practice (≥1.3 dataframe-protocol), but the contract is undocumented and the pandas-only sanitisation path is silently bypassed. Decision: keep pandas conversion + sanitise pre-fit.
    """
    import json
    manifest_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "audit", "critique_2026_05_24",
        "manifests", "DONE_w16b-bruteforce-pysr-polars.json",
    )
    manifest_path = os.path.normpath(manifest_path)
    assert os.path.exists(manifest_path), f"Wave 16B manifest missing: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    assert "pysr_polars_native" in manifest, "manifest must record pysr_polars_native: bool"
    assert isinstance(manifest["pysr_polars_native"], bool)
    assert "pysr_finding_note" in manifest, "manifest must record pysr_finding_note: str"
    note = manifest["pysr_finding_note"]
    assert isinstance(note, str) and len(note) >= 50, (
        "pysr_finding_note must be substantive (>=50 chars), not a stub"
    )
