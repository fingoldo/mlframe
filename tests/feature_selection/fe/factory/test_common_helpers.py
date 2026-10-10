"""Shared factory helpers: output paths, quantile-binned MI, the MAE / RMSE downstream harness and the brainstorm run_case / agg10 consumers of it."""

import json
import sys

import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.fe_operator_factory.brainstorm.h import run_case
from mlframe.feature_selection._benchmarks.fe_operator_factory.brainstorm.ops_pair import CASES
from mlframe.feature_selection._benchmarks.fe_operator_factory.common import _paths
from mlframe.feature_selection._benchmarks.fe_operator_factory.common._paths import data_dir, results_dir, scratch_dir
from mlframe.feature_selection._benchmarks.fe_operator_factory.common.binning import mi, mi_b, qbin
from mlframe.feature_selection._benchmarks.fe_operator_factory.common.downstream import (
    errors_by_feature_set,
    fit_errors,
    make_models,
    rel_improvement,
    relative_table,
)
from mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study import agg10


def test_scratch_dir_is_outside_the_repo_and_data_dir_follows_the_fresh_flag(tmp_path, monkeypatch):
    """Scratch output goes to the (patched) scratch root; reads use the committed results unless ``--fresh`` is on the command line."""
    monkeypatch.setattr(_paths, "SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["agg"])
    assert scratch_dir("stat_study") == tmp_path / "stat_study"
    assert data_dir("stat_study") == results_dir("stat_study")
    assert results_dir("stat_study").is_dir()
    monkeypatch.setattr(sys, "argv", ["agg", "--fresh"])
    assert data_dir("stat_study") == tmp_path / "stat_study"
    assert _paths.PKG_DIR not in tmp_path.parents


def test_qbin_is_equal_frequency_and_mi_detects_dependence():
    """qbin splits into equal-size bins; MI of a monotone transform with the target is high, of independent noise near zero."""
    rng = np.random.default_rng(0)
    x = rng.random(5000)
    assert np.bincount(qbin(x, 10)).tolist() == [500] * 10
    yb = qbin(x**3, 10)
    assert mi(x, yb) > 2.0
    assert mi(rng.random(5000), yb) < 0.02
    assert mi_b(yb, yb) == pytest.approx(np.log(10), abs=1e-9)


def test_rel_improvement_sign_convention():
    """Lower error than the baseline is a positive improvement; a zero baseline gives NaN rather than a division error."""
    assert rel_improvement(0.8, 1.0) == pytest.approx(0.2)
    assert rel_improvement(1.2, 1.0) == pytest.approx(-0.2)
    assert np.isnan(rel_improvement(1.0, 0.0))


def test_fit_errors_report_mae_rmse_and_reference_r2():
    """fit_errors reports MAE and RMSE of the held-out part, plus R^2 as a reference value, all computed by hand here."""
    rng = np.random.default_rng(0)
    X = rng.random((400, 2))
    y = 3 * X[:, 0] + 0.1 * rng.standard_normal(400)
    model = make_models()["ridge"]
    e = fit_errors(model, X[:300], y[:300], X[300:], y[300:])
    resid = y[300:] - model.predict(X[300:])
    assert set(e) == {"mae", "rmse", "r2"}
    assert e["mae"] == pytest.approx(np.abs(resid).mean())
    assert e["rmse"] == pytest.approx(np.sqrt((resid**2).mean()))
    assert e["r2"] == pytest.approx(1 - (resid**2).sum() / ((y[300:] - y[300:].mean()) ** 2).sum())


def test_relative_table_uses_the_raw_row_of_the_same_model():
    """A feature that carries the signal beats the raw columns for both models; the baseline row itself is exactly 0."""
    rng = np.random.default_rng(1)
    x, z = rng.random((2, 2000))
    y = x * z + 0.05 * rng.standard_normal(2000)
    raw = np.column_stack([x, z])
    sets = {"raw": (raw[:1500], raw[1500:]), "+prod": (np.column_stack([raw, x * z])[:1500], np.column_stack([raw, x * z])[1500:])}
    rel = relative_table(errors_by_feature_set(sets, y[:1500], y[1500:]))
    assert rel["ridge|raw"] == {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
    assert rel["ridge|+prod"]["mae"] > 0.3
    assert rel["ridge|+prod"]["rmse"] > 0.3
    assert rel["ridge|+prod"]["r2"] > 0.0


def test_run_case_reports_mae_rmse_and_reference_r2(capsys):
    """The brainstorm harness records relative MAE / RMSE (decisions) and a reference R^2 delta for ridge and HGB, and a warp feature helps ridge on the W target."""
    gen, new_fn = CASES["C_W"]
    mean = run_case("C_W", gen, new_fn, [0, 1], seeds=1, n=2000, pairs=False)
    assert "rel_mae|ridge|new" in mean and "rel_rmse|hgb|new" in mean
    assert "rel_r2|ridge|new" in mean
    assert mean["rel_mae|ridge|new"] > 0.2
    assert "ref. R2 delta" in capsys.readouterr().out


def test_agg10_computes_relative_improvement_and_skips_legacy(tmp_path, monkeypatch, capsys):
    """agg10 prints relative MAE / RMSE improvement from new-format records and counts the legacy R^2-only ones as skipped."""
    monkeypatch.setattr(_paths, "SCRATCH_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["agg10", "--fresh"])
    sub = scratch_dir("stat_study")
    keys = ["raw", "+preset", "+p1", "+p2", "+p1+p2"]
    err = {f"{m}|{k}": (1.0 if k == "raw" else 0.5) for m in ("ridge", "hgb") for k in keys}
    rec = {"target": "T", "n": 100, "seed": 0, "mae": err, "rmse": err}
    legacy = {"target": "T", "n": 100, "seed": 0, "r2": {}}
    (sub / "ds_0.json").write_text(json.dumps([rec, legacy]))
    agg10.main()
    out = capsys.readouterr().out
    assert "1 records with MAE/RMSE, 1 legacy" in out
    assert "0.500" in out
