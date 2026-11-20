"""Regression tests for the adaptive-degree / adaptive-arity fit axis, and the CMIM / copula / adaptive-arity device fallback logs."""
import logging
import sys
import types

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_adaptive_arity_fe as arity_mod
from mlframe.feature_selection.filters import _orthogonal_cmim_fe as cmim_mod
from mlframe.feature_selection.filters import _orthogonal_copula_gpu as copula_gpu
from mlframe.feature_selection.filters._orthogonal_adaptive_degree_fe import hybrid_orth_mi_adaptive_degree_fe_with_recipes
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _evaluate_basis_column


def _nan_frame(n: int = 3000):
    """A frame whose first column carries a quadratic signal and 10% NaN, plus a noise column and a binary target."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    y = (x**2 > 1).astype(int)
    xa = x.copy()
    xa[::10] = np.nan
    return pd.DataFrame({"a": xa, "b": rng.normal(size=n)}), y


def test_adaptive_degree_recipe_params_finite_for_nan_source():
    """The frozen preprocess params of a NaN-bearing source are finite and the emitted column equals the replay on the finite rows."""
    X, y = _nan_frame()
    _, scores, recipes = hybrid_orth_mi_adaptive_degree_fe_with_recipes(X, y)
    rec = {r.name: r for r in recipes}
    name = "a__" + scores.loc[scores.source_col == "a", "engineered_col"].iloc[0].split("__", 1)[1]
    pp = rec[name].extra["preprocess_params"]
    assert np.isfinite(pp["mean"]) and np.isfinite(pp["std"])
    row = scores[scores.engineered_col == name].iloc[0]
    col = np.asarray(X["a"].to_numpy(), dtype=np.float64)
    fin = np.isfinite(col)
    replay = _evaluate_basis_column(np.where(fin, col, pp["mean"]), str(row["basis"]), int(row["degree"]), preprocess_params=pp)
    X_aug, _, _ = hybrid_orth_mi_adaptive_degree_fe_with_recipes(X, y)
    np.testing.assert_allclose(np.asarray(X_aug[name])[fin], replay[fin], rtol=1e-6, atol=1e-6)


def test_adaptive_arity_freeze_leg_params_finite_for_nan_source():
    """The arity family freezes finite params for a NaN-bearing leg."""
    X, y = _nan_frame()
    X["c"] = X["b"] * 0.5 + np.random.default_rng(1).normal(size=len(X))
    out = arity_mod.hybrid_orth_mi_adaptive_arity_fe_with_recipes(X, y, cols=["a", "b", "c"], top_count=3, adaptive_min_uplift=0.0, adaptive_min_abs_mi_frac=0.0)
    recipes = out[3]
    pps = [v for r in recipes for k, v in r.extra.items() if k.startswith("preprocess_params") and v]
    assert pps
    for pp in pps:
        assert np.isfinite(pp["mean"]) and np.isfinite(pp["std"])


def test_cmim_device_scorer_warns_on_out_of_range_codes(monkeypatch, caplog):
    """A device-path decline on out-of-range candidate codes is a WARNING, not a debug line."""
    from mlframe.feature_selection.filters import _gpu_policy, _scorer_device_dispatch

    monkeypatch.setattr(_gpu_policy, "cuda_available_for_run", lambda: True)
    monkeypatch.setattr(_scorer_device_dispatch, "scorer_device_choice", lambda *a, **k: True)
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(asarray=np.asarray))
    eng = np.array([[0, 1, 99, 2]], dtype=np.int16)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    with caplog.at_level(logging.WARNING, logger=cmim_mod.logger.name):
        got = cmim_mod._cmim_scores_device(eng, y, [np.array([0, 1, 1, 0])], ["z"], ["x"], 10)
    assert got is None
    assert any("candidate codes outside" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_adaptive_arity_resident_block_failure_warns(monkeypatch, caplog):
    """A failing GPU-resident MI path logs a WARNING before the host scorer takes over."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_pair_cross_fe

    monkeypatch.setattr(_orth_pair_cross_fe, "_crossbasis_device_born_on", lambda: True)
    monkeypatch.setitem(sys.modules, "cupy", None)
    with caplog.at_level(logging.WARNING, logger=arity_mod.logger.name):
        got = arity_mod._adaptive_arity_mi_resident_block(pd.DataFrame({"a": [1.0]}), np.array([0]), [(("a", "b"), (1, 1), "n", np.zeros(1))], basis="hermite", nbins=10)
    assert got is None
    assert any("GPU-resident MI failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_copula_device_rejects_out_of_range_target_codes(caplog):
    """Target bin codes beyond the table are refused on the host before any kernel indexes a histogram with them."""
    cols = np.random.default_rng(0).normal(size=(3, 50)).astype(np.float32)
    vi = np.arange(50) % 25
    with caplog.at_level(logging.WARNING, logger=copula_gpu.logger.name):
        assert copula_gpu.copula_columns_device(cols, vi, 20) is None
    assert any("target bin codes outside" in r.getMessage() for r in caplog.records)


def test_copula_column_block_respects_cuda_grid_row_limit():
    """A block never exceeds the 65535 grid rows a launch can address, however much memory is free."""
    assert copula_gpu.column_block(10**6, 1e12, 100) == 65535
    assert copula_gpu.column_block(10, 0.0, 100) == 1
    assert copula_gpu.column_block(5, 1e12, 100) == 5
