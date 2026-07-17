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

from types import SimpleNamespace

from mlframe.reporting.diagnostics_dispatch import (
    build_combined_html_report,
    render_calibration_drift_diagnostic,
    render_decision_curve_diagnostic,
    render_model_comparison_diagnostic,
    render_model_comparison_from_suite,
    render_pdp_ice_diagnostic,
    render_shap_diagnostic,
    render_slice_finder_diagnostic,
    render_target_acf_diagnostic,
)

PNG = "matplotlib[png]"


def _png_exists(base: str) -> bool:
    """Helper: Png exists."""
    return os.path.exists(base + ".png") or os.path.exists(base + ".matplotlib.png")


@pytest.fixture
def binary_frame():
    """Binary frame."""
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
    """Helper: Fit tree."""
    from sklearn.ensemble import RandomForestClassifier

    m = RandomForestClassifier(n_estimators=12, random_state=0)
    m.fit(df.values, y)
    return m


def test_pdp_ice_renders_default_on(tmp_path, binary_frame):
    """Pdp ice renders default on."""
    df, y, _score, _ts = binary_frame
    model = _fit_tree(df, y)
    base = str(tmp_path / "m")
    md = {}
    ok = render_pdp_ice_diagnostic(
        model=model,
        df=df,
        feature_names=list(df.columns),
        feature_importances=list(model.feature_importances_),
        plot_outputs=PNG,
        base_path=base,
        metrics_dict=md,
        top_features=2,
    )
    assert ok and _png_exists(base + "_pdp_ice")
    assert "pdp_ice" in md["charts"]["saved"]
    assert (base + "_pdp_ice") in md["charts"]["paths"]


def test_pdp_ice_skips_without_model(tmp_path, binary_frame):
    """Pdp ice skips without model."""
    df, _y, _s, _ts = binary_frame
    md = {}
    assert (
        render_pdp_ice_diagnostic(
            model=None,
            df=df,
            feature_names=list(df.columns),
            feature_importances=None,
            plot_outputs=PNG,
            base_path=str(tmp_path / "m"),
            metrics_dict=md,
        )
        is False
    )


def test_slice_finder_renders_default_on(tmp_path, binary_frame):
    """Slice finder renders default on."""
    df, y, score, _ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    ok = render_slice_finder_diagnostic(
        df=df,
        y_true=y,
        y_pred=score,
        task="classification",
        feature_names=list(df.columns),
        plot_outputs=PNG,
        base_path=base,
        metrics_dict=md,
    )
    assert ok and _png_exists(base + "_weak_slices")
    assert "weak_slices" in md["charts"]["saved"]
    assert "weak_slices" in md  # the table surfaced


def test_decision_curve_renders_default_on(tmp_path, binary_frame):
    """Decision curve renders default on."""
    _df, y, score, _ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    ok = render_decision_curve_diagnostic(
        y_true=y,
        y_score=score,
        plot_outputs=PNG,
        base_path=base,
        metrics_dict=md,
    )
    assert ok and _png_exists(base + "_decision_curve")
    assert "decision_curve" in md["charts"]["saved"]
    assert "decision_curve_useful" in md


def test_calibration_drift_renders_default_on(tmp_path, binary_frame):
    """Calibration drift renders default on."""
    _df, y, score, ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    ok = render_calibration_drift_diagnostic(
        y_true=y,
        y_score=score,
        timestamps=ts,
        plot_outputs=PNG,
        base_path=base,
        metrics_dict=md,
    )
    assert ok and _png_exists(base + "_calibration_drift")
    assert "calibration_drift" in md["charts"]["saved"]


def test_calibration_drift_skips_without_timestamps(tmp_path, binary_frame):
    """Calibration drift skips without timestamps."""
    _df, y, score, _ts = binary_frame
    md = {}
    assert (
        render_calibration_drift_diagnostic(
            y_true=y,
            y_score=score,
            timestamps=None,
            plot_outputs=PNG,
            base_path=str(tmp_path / "m"),
            metrics_dict=md,
        )
        is False
    )


def test_target_acf_renders_default_on(tmp_path, binary_frame):
    """Target acf renders default on."""
    _df, y, _score, ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    ok = render_target_acf_diagnostic(
        y_true=y.astype(float),
        timestamps=ts,
        plot_outputs=PNG,
        base_path=base,
        metrics_dict=md,
    )
    assert ok and _png_exists(base + "_target_acf")
    assert "target_acf" in md["charts"]["saved"]


def test_shap_default_on_for_tree(tmp_path, binary_frame):
    """Shap default on for tree."""
    pytest.importorskip("shap")
    df, y, _score, _ts = binary_frame
    model = _fit_tree(df, y)
    base = str(tmp_path / "m")
    md = {}
    ok = render_shap_diagnostic(
        model=model,
        df=df,
        feature_names=list(df.columns),
        plot_outputs=PNG,
        base_path=base,
        metrics_dict=md,
        max_rows=400,
        top_k=3,
    )
    assert ok
    assert "shap_panels" in md["charts"]["saved"]
    assert md["charts"].get("paths")


def test_shap_opt_in_for_non_tree(tmp_path, binary_frame):
    """Shap opt in for non tree."""
    pytest.importorskip("shap")
    from sklearn.linear_model import LogisticRegression

    df, y, _score, _ts = binary_frame
    model = LogisticRegression(max_iter=200).fit(df.values, y)
    base = str(tmp_path / "m")
    md = {}
    # Non-tree, kernel disabled -> skip cheaply.
    assert (
        render_shap_diagnostic(
            model=model,
            df=df,
            feature_names=list(df.columns),
            plot_outputs=PNG,
            base_path=base,
            metrics_dict=md,
            allow_kernel=False,
        )
        is False
    )


def test_model_comparison_default_on_with_two_models(tmp_path, binary_frame):
    """Model comparison default on with two models."""
    _df, y, score, _ts = binary_frame
    rng = np.random.default_rng(1)
    weak = np.clip(score + rng.normal(0, 0.3, len(score)), 0, 1)
    per_model = {
        "strong": {"y_true": y, "y_score": score, "metrics": {"roc_auc": 0.85}},
        "weak": {"y_true": y, "y_score": weak, "metrics": {"roc_auc": 0.60}},
    }
    base = str(tmp_path / "cmp")
    md = {}
    ok = render_model_comparison_diagnostic(
        per_model=per_model,
        task_type="binary",
        plot_outputs=PNG,
        base_path=base,
        metrics_dict=md,
    )
    assert ok and _png_exists(base + "_model_comparison")
    assert "model_comparison" in md["charts"]["saved"]


def test_model_comparison_skips_single_model(tmp_path, binary_frame):
    """Model comparison skips single model."""
    _df, y, score, _ts = binary_frame
    md = {}
    assert (
        render_model_comparison_diagnostic(
            per_model={"only": {"y_true": y, "y_score": score, "metrics": {}}},
            task_type="binary",
            plot_outputs=PNG,
            base_path=str(tmp_path / "cmp"),
            metrics_dict=md,
        )
        is False
    )


def _suite_entry(y, score, auc):
    """Mirror the suite's per-model SimpleNamespace record shape (test_target / test_probs / nested metrics)."""
    probs = np.column_stack([1.0 - score, score])
    return SimpleNamespace(
        model=SimpleNamespace(),
        test_target=y,
        test_probs=probs,
        test_preds=None,
        metrics={"test": {1: {"roc_auc": auc}, "charts": {"saved": []}}},
    )


def test_model_comparison_from_suite_renders_for_two_models(tmp_path, binary_frame):
    """Model comparison from suite renders for two models."""
    _df, y, score, _ts = binary_frame
    rng = np.random.default_rng(2)
    weak = np.clip(score + rng.normal(0, 0.3, len(score)), 0, 1)
    entries = [
        _suite_entry(y, score, 0.85),
        _suite_entry(y, weak, 0.60),
    ]
    base = str(tmp_path / "tgt")
    md = {}
    ok = render_model_comparison_from_suite(
        model_entries=entries,
        target_type="binary_classification",
        plot_outputs=PNG,
        base_path=base,
        metrics_dict=md,
    )
    assert ok and _png_exists(base + "_model_comparison")
    assert "model_comparison" in md["charts"]["saved"]


def test_model_comparison_from_suite_skips_single_model(tmp_path, binary_frame):
    """Model comparison from suite skips single model."""
    _df, y, score, _ts = binary_frame
    md = {}
    assert (
        render_model_comparison_from_suite(
            model_entries=[_suite_entry(y, score, 0.85)],
            target_type="binary_classification",
            plot_outputs=PNG,
            base_path=str(tmp_path / "tgt"),
            metrics_dict=md,
        )
        is False
    )


def test_combined_html_stitches_rendered_charts(tmp_path, binary_frame):
    """Combined html stitches rendered charts."""
    _df, y, score, ts = binary_frame
    base = str(tmp_path / "m")
    md = {}
    render_decision_curve_diagnostic(y_true=y, y_score=score, plot_outputs=PNG, base_path=base, metrics_dict=md)
    render_calibration_drift_diagnostic(y_true=y, y_score=score, timestamps=ts, plot_outputs=PNG, base_path=base, metrics_dict=md)
    paths = md["charts"]["paths"]
    out = build_combined_html_report(
        base_path=base,
        chart_paths=paths,
        plot_outputs=PNG,
        title="m report",
        metrics_dict=md,
    )
    assert out and os.path.exists(out)
    assert md["charts"].get("combined_report") == out
    html = open(out, encoding="utf-8").read()
    assert "m report" in html and "decision_curve" in html


def test_combined_html_orders_weak_slices_before_weak_segments(tmp_path):
    """Combined html orders weak slices before weak segments."""
    base = str(tmp_path / "m")
    # Recorded order puts the per-split weak-segment heatmap first (it renders in the metrics phase, before the
    # once-on-test worst-slices chart). The combined report must reorder so worst-slices leads.
    paths = [base + "_weak_segments", base + "_error_bias", base + "_weak_slices", base + "_pdp_ice"]
    for p in paths:
        open(p + ".png", "w").close()
    md = {}
    out = build_combined_html_report(base_path=base, chart_paths=paths, plot_outputs=PNG, title="m report", metrics_dict=md)
    assert out and os.path.exists(out)
    html = open(out, encoding="utf-8").read()
    assert html.index("m_weak_slices") < html.index("m_weak_segments")
