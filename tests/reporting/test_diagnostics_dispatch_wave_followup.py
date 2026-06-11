"""Unit + render coverage for the wave-followup standalone diagnostic orchestrators.

Each ``render_*_diagnostic`` in ``reporting.diagnostics_dispatch`` is exercised in isolation: it renders a file
default-on (or opt-in when costly), records the artifact in the ``charts`` accounting, and skips cheaply when its
inputs are absent. These guard the orchestration layer independently of a full suite run.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.diagnostics_dispatch import (
    build_combined_html_report,
    render_calibration_drift_diagnostic,
    render_decision_curve_diagnostic,
    render_model_comparison_diagnostic,
    render_pdp_ice_diagnostic,
    render_shap_diagnostic,
    render_slice_finder_diagnostic,
    render_target_acf_diagnostic,
)

PNG = "matplotlib[png]"


def _png_exists(base: str) -> bool:
    return os.path.exists(base + ".png") or os.path.exists(base + ".matplotlib.png")


@pytest.fixture
def binary_frame():
    rng = np.random.default_rng(0)
    n = 800
    f0 = rng.uniform(0, 1, n)
    f1 = rng.uniform(0, 1, n)
    f2 = rng.normal(0, 1, n)
    df = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    logit = 3 * f0 - 1.5 + 0.4 * f2
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    score = 1 / (1 + np.exp(-(2.5 * f0 - 1.2)))
    ts = np.arange(n)
    return df, y, score, ts


def _fit_tree(df, y):
    from sklearn.ensemble import RandomForestClassifier

    m = RandomForestClassifier(n_estimators=12, random_state=0)
    m.fit(df.values, y)
    return m


def test_pdp_ice_renders_default_on(tmp_path, binary_frame):
    df, y, _score, _ts = binary_frame
    model = _fit_tree(df, y)
    base = str(tmp_path / "m")
    md = {}
    ok = render_pdp_ice_diagnostic(
        model=model, df=df, feature_names=list(df.columns),
        feature_importances=list(model.feature_importances_),
        plot_outputs=PNG, base_path=base, metrics_dict=md, top_features=2,
    )
    assert ok and _png_exists(base + "_pdp_ice")
    assert "pdp_ice" in md["charts"]["saved"]
    assert (base + "_pdp_ice") in md["charts"]["paths"]


def test_pdp_ice_skips_without_model(tmp_path, binary_frame):
    df, _y, _s, _ts = binary_frame
    md = {}
    assert render_pdp_ice_diagnostic(
        model=None, df=df, feature_names=list(df.columns), feature_importances=None,
        plot_outputs=PNG, base_path=str(tmp_path / "m"), metrics_dict=md,
    ) is False


def test_slice_finder_renders_default_on(tmp_path, binary_frame):
    df, y, score, _ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    ok = render_slice_finder_diagnostic(
        df=df, y_true=y, y_pred=score, task="classification",
        feature_names=list(df.columns), plot_outputs=PNG, base_path=base, metrics_dict=md,
    )
    assert ok and _png_exists(base + "_weak_slices")
    assert "weak_slices" in md["charts"]["saved"]
    assert "weak_slices" in md  # the table surfaced


def test_decision_curve_renders_default_on(tmp_path, binary_frame):
    df, y, score, _ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    ok = render_decision_curve_diagnostic(
        y_true=y, y_score=score, plot_outputs=PNG, base_path=base, metrics_dict=md,
    )
    assert ok and _png_exists(base + "_decision_curve")
    assert "decision_curve" in md["charts"]["saved"]
    assert "decision_curve_useful" in md


def test_calibration_drift_renders_default_on(tmp_path, binary_frame):
    df, y, score, ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    ok = render_calibration_drift_diagnostic(
        y_true=y, y_score=score, timestamps=ts, plot_outputs=PNG, base_path=base, metrics_dict=md,
    )
    assert ok and _png_exists(base + "_calibration_drift")
    assert "calibration_drift" in md["charts"]["saved"]


def test_calibration_drift_skips_without_timestamps(tmp_path, binary_frame):
    df, y, score, _ts = binary_frame
    md = {}
    assert render_calibration_drift_diagnostic(
        y_true=y, y_score=score, timestamps=None, plot_outputs=PNG,
        base_path=str(tmp_path / "m"), metrics_dict=md,
    ) is False


def test_target_acf_renders_default_on(tmp_path, binary_frame):
    df, y, _score, ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    ok = render_target_acf_diagnostic(
        y_true=y.astype(float), timestamps=ts, plot_outputs=PNG, base_path=base, metrics_dict=md,
    )
    assert ok and _png_exists(base + "_target_acf")
    assert "target_acf" in md["charts"]["saved"]


def test_shap_default_on_for_tree(tmp_path, binary_frame):
    pytest.importorskip("shap")
    df, y, _score, _ts = binary_frame
    model = _fit_tree(df, y)
    base = str(tmp_path / "m")
    md = {}
    ok = render_shap_diagnostic(
        model=model, df=df, feature_names=list(df.columns),
        plot_outputs=PNG, base_path=base, metrics_dict=md, max_rows=400, top_k=3,
    )
    assert ok
    assert "shap_panels" in md["charts"]["saved"]
    assert md["charts"].get("paths")


def test_shap_opt_in_for_non_tree(tmp_path, binary_frame):
    pytest.importorskip("shap")
    from sklearn.linear_model import LogisticRegression

    df, y, _score, _ts = binary_frame
    model = LogisticRegression(max_iter=200).fit(df.values, y)
    base = str(tmp_path / "m")
    md = {}
    # Non-tree, kernel disabled -> skip cheaply.
    assert render_shap_diagnostic(
        model=model, df=df, feature_names=list(df.columns),
        plot_outputs=PNG, base_path=base, metrics_dict=md, allow_kernel=False,
    ) is False


def test_model_comparison_default_on_with_two_models(tmp_path, binary_frame):
    df, y, score, _ts = binary_frame
    rng = np.random.default_rng(1)
    weak = np.clip(score + rng.normal(0, 0.3, len(score)), 0, 1)
    per_model = {
        "strong": {"y_true": y, "y_score": score, "metrics": {"roc_auc": 0.85}},
        "weak": {"y_true": y, "y_score": weak, "metrics": {"roc_auc": 0.60}},
    }
    base = str(tmp_path / "cmp")
    md = {}
    ok = render_model_comparison_diagnostic(
        per_model=per_model, task_type="binary", plot_outputs=PNG, base_path=base, metrics_dict=md,
    )
    assert ok and _png_exists(base + "_model_comparison")
    assert "model_comparison" in md["charts"]["saved"]


def test_model_comparison_skips_single_model(tmp_path, binary_frame):
    df, y, score, _ts = binary_frame
    md = {}
    assert render_model_comparison_diagnostic(
        per_model={"only": {"y_true": y, "y_score": score, "metrics": {}}},
        task_type="binary", plot_outputs=PNG, base_path=str(tmp_path / "cmp"), metrics_dict=md,
    ) is False


def test_combined_html_stitches_rendered_charts(tmp_path, binary_frame):
    df, y, score, ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    render_decision_curve_diagnostic(y_true=y, y_score=score, plot_outputs=PNG, base_path=base, metrics_dict=md)
    render_calibration_drift_diagnostic(y_true=y, y_score=score, timestamps=ts, plot_outputs=PNG, base_path=base, metrics_dict=md)
    paths = md["charts"]["paths"]
    out = build_combined_html_report(
        base_path=base, chart_paths=paths, plot_outputs=PNG, title="m report", metrics_dict=md,
    )
    assert out and os.path.exists(out)
    assert md["charts"].get("combined_report") == out
    html = open(out, encoding="utf-8").read()
    assert "m report" in html and "decision_curve" in html
