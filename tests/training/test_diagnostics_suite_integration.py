"""End-to-end: the wave-4 diagnostic charts render DEFAULT-ON through a real
``train_mlframe_models_suite`` run and a run can assert chart presence.

A tiny suite (HGB, few iterations) is trained for binary classification and for
regression with charts saved + a timestamp column, then we assert:
  * binary: the binary curve-panel grid file (ROC/PR/...) is saved (INV-10);
  * error-analysis files (weak_segments / error_bias) land for both task types;
  * per-target diagnostics (target distribution overlay, adversarial) land;
  * the temporal panels (residual_vs_time / metric_over_time) land when timestamps
    cover the split (INV-9 / INV-26);
  * a ``charts`` accounting key with a non-empty ``saved`` list is recorded (INV-48).

These guard against a future silent revert of the default-ON wiring.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

from mlframe.training.configs import ReportingConfig, TargetTypes
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import OutputConfig
from mlframe.training.diagnostics import LearningCurveConfig

from .shared import SimpleFeaturesAndTargetsExtractor, get_cpu_config, skip_if_dependency_missing


def _make_frame(n: int, *, binary: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0, 1, n)
    f1 = rng.uniform(0, 1, n)
    f2 = rng.normal(0, 1, n)
    df = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    if binary:
        logit = 3 * f0 - 1.5 + 0.5 * f2
        df["target"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    else:
        df["target"] = 2 * f0 + 0.5 * f2 + rng.normal(0, 0.2, n)
    # Monotone timestamps so the temporal panels have a real time axis.
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="h")
    return df


def _saved_chart_files(data_dir: str):
    return [os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "charts", "**", "*"), recursive=True) if os.path.isfile(p)]


def _collect_charts_acc(obj):
    """Walk metadata / model metrics dicts for any ``charts`` accounting dict and merge the saved lists."""
    saved = []

    def _walk(o):
        if isinstance(o, dict):
            c = o.get("charts")
            if isinstance(c, dict) and isinstance(c.get("saved"), list):
                saved.extend(c["saved"])
            for v in o.values():
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)

    _walk(obj)
    return saved


@pytest.fixture
def reporting_cfg():
    # Suppress the legacy per-class calibration matplotlib figure + FI plot; the
    # new diagnostics still render default-ON via the DSL plot_outputs path.
    return ReportingConfig(show_perf_chart=False, show_fi=False, plot_outputs="matplotlib[png]")


def test_binary_suite_renders_diagnostics_default_on(tmp_path, reporting_cfg):
    skip_if_dependency_missing("hgb")
    df = _make_frame(900, binary=True)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False, ts_field="timestamp")
    data_dir = str(tmp_path)
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="diag_bin",
        model_name="hgb_bin",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        reporting_config=reporting_cfg,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models", save_charts=True),
        verbose=0,
    )
    assert TargetTypes.BINARY_CLASSIFICATION in models
    files = _saved_chart_files(data_dir)
    joined = " ".join(files)
    # INV-10: binary curve panels rendered by default.
    assert "binary_panels" in joined, f"binary panel grid not saved; files={files}"
    # INV-23 / R-2 / R-8: per-split error-analysis charts.
    assert "weak_segments" in joined, f"weak-segment heatmap not saved; files={files}"
    assert "error_bias" in joined, f"error-bias chart not saved; files={files}"
    # INV-11 / R-3 + R-1: per-target distribution overlay + adversarial.
    assert "target_dist" in joined, f"target distribution overlay not saved; files={files}"
    # Wave A+B default-on diagnostics: PDP/ICE, slice-finder, decision-curve render by default; SHAP for the tree
    # model; calibration-drift + target-ACF when timestamps cover the split (the frame carries a monotone timestamp).
    assert "pdp_ice" in joined, f"PDP/ICE not saved; files={files}"
    assert "weak_slices" in joined, f"slice-finder not saved; files={files}"
    assert "decision_curve" in joined, f"decision-curve not saved; files={files}"
    assert "shap" in joined, f"SHAP panels not saved for tree model; files={files}"
    assert "calibration_drift" in joined, f"calibration-drift not saved; files={files}"
    assert "target_acf" in joined, f"target ACF/PACF not saved; files={files}"
    # Combined single-page HTML index stitched from the artifacts.
    html = [f for f in files if f.endswith("_report.html")]
    assert html, f"combined HTML report not saved; files={files}"
    # INV-48: chart accounting recorded somewhere in the returned artifacts.
    saved_acc = _collect_charts_acc(metadata) + _collect_charts_acc(models)
    assert saved_acc, "no charts accounting recorded in metadata/models"


def test_per_model_calibration_chart_is_written_without_missing_dir(tmp_path):
    """Regression: ``plot_file`` reaches ``_compute_split_metrics`` as a filename prefix ending in the model-type name
    (not a directory ending in os.sep). The split-chart path is derived from that prefix and its directory must exist
    before ``show_calibration_plot`` savefigs the reliability diagram, otherwise the suite dies with FileNotFoundError.
    Enable the per-class reliability chart so the exact savefig path is exercised."""
    skip_if_dependency_missing("hgb")
    df = _make_frame(600, binary=True)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False, ts_field="timestamp")
    data_dir = str(tmp_path)
    models, _ = train_mlframe_models_suite(
        df=df,
        target_name="diag_bin",
        model_name="hgb_bin",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        reporting_config=ReportingConfig(show_perf_chart=True, show_fi=False, plot_outputs="matplotlib[png]"),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models", save_charts=True),
        verbose=0,
    )
    assert TargetTypes.BINARY_CLASSIFICATION in models
    perfplots = [
        p for p in glob.glob(os.path.join(data_dir, "charts", "**", "*perfplot*.png"), recursive=True) if os.path.isfile(p)
    ]
    assert perfplots, "per-model calibration perfplot.png was not saved (chart dir not created before savefig)"


def test_regression_suite_renders_diagnostics_default_on(tmp_path, reporting_cfg):
    skip_if_dependency_missing("hgb")
    df = _make_frame(900, binary=False)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True, ts_field="timestamp")
    data_dir = str(tmp_path)
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="diag_reg",
        model_name="hgb_reg",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        reporting_config=reporting_cfg,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models", save_charts=True),
        verbose=0,
    )
    assert TargetTypes.REGRESSION in models
    files = _saved_chart_files(data_dir)
    joined = " ".join(files)
    assert "weak_segments" in joined, f"weak-segment heatmap not saved; files={files}"
    assert "error_bias" in joined, f"error-bias chart not saved; files={files}"
    assert "target_dist" in joined, f"target distribution overlay not saved; files={files}"
    # INV-26 / INV-9: temporal residual + metric panels when timestamps cover the split.
    assert "residual_vs_time" in joined, f"residual-vs-time not saved; files={files}"
    # Wave A+B default-on diagnostics for regression: PDP/ICE, slice-finder, SHAP (tree), target-ACF (timestamps).
    assert "pdp_ice" in joined, f"PDP/ICE not saved; files={files}"
    assert "weak_slices" in joined, f"slice-finder not saved; files={files}"
    assert "shap" in joined, f"SHAP panels not saved for tree model; files={files}"
    assert "target_acf" in joined, f"target ACF/PACF not saved; files={files}"
    html = [f for f in files if f.endswith("_report.html")]
    assert html, f"combined HTML report not saved; files={files}"
    saved_acc = _collect_charts_acc(metadata) + _collect_charts_acc(models)
    assert saved_acc, "no charts accounting recorded in metadata/models"


def _make_multiclass_frame(n: int, *, n_classes: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0, 1, n)
    f1 = rng.uniform(0, 1, n)
    f2 = rng.normal(0, 1, n)
    df = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    score = 2 * f0 + 0.5 * f2 + rng.normal(0, 0.3, n)
    df["target"] = pd.qcut(score, n_classes, labels=False).astype(int)
    return df


def test_multiclass_suite_renders_diagnostics_default_on(tmp_path, reporting_cfg):
    skip_if_dependency_missing("hgb")
    df = _make_multiclass_frame(900, n_classes=3)
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column="target", regression=False, target_type=TargetTypes.MULTICLASS_CLASSIFICATION,
    )
    data_dir = str(tmp_path)
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="diag_mc",
        model_name="hgb_mc",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        reporting_config=reporting_cfg,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models", save_charts=True),
        verbose=0,
    )
    assert TargetTypes.MULTICLASS_CLASSIFICATION in models
    files = _saved_chart_files(data_dir)
    joined = " ".join(files)
    assert "multiclass_panels" in joined, f"multiclass panel grid not saved; files={files}"
    # Model/preds-based diagnostics default-on regardless of class count.
    assert "pdp_ice" in joined, f"PDP/ICE not saved; files={files}"
    assert "weak_slices" in joined, f"slice-finder not saved; files={files}"
    assert "shap" in joined, f"SHAP panels not saved for tree model; files={files}"
    saved_acc = _collect_charts_acc(metadata) + _collect_charts_acc(models)
    assert saved_acc, "no charts accounting recorded in metadata/models"


def _make_multilabel_frame(n: int, *, n_labels: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0, 1, n)
    f1 = rng.uniform(0, 1, n)
    f2 = rng.normal(0, 1, n)
    df = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    labels = []
    for i in range(n):
        l0 = int(f0[i] > 0.5)
        l1 = int(f1[i] > 0.4)
        l2 = int(f2[i] > 0.0)
        labels.append([l0, l1, l2][:n_labels])
    df["target"] = pd.Series(labels, dtype=object)
    return df


def test_multilabel_suite_renders_threshold_sweep_default_on(tmp_path, reporting_cfg):
    skip_if_dependency_missing("hgb")
    df = _make_multilabel_frame(900, n_labels=3)
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column="target", regression=False, target_type=TargetTypes.MULTILABEL_CLASSIFICATION,
    )
    data_dir = str(tmp_path)
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="diag_ml",
        model_name="hgb_ml",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        reporting_config=reporting_cfg,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models", save_charts=True),
        verbose=0,
    )
    assert TargetTypes.MULTILABEL_CLASSIFICATION in models
    files = _saved_chart_files(data_dir)
    joined = " ".join(files)
    # The multilabel panel grid carries THRESHOLD_SWEEP default-on (rendered into ``*_multilabel_panels.*``).
    assert "multilabel_panels" in joined, f"multilabel panel grid not saved; files={files}"
    saved_acc = _collect_charts_acc(metadata) + _collect_charts_acc(models)
    assert saved_acc, "no charts accounting recorded in metadata/models"


def test_learning_curve_opt_in_when_enabled(tmp_path):
    """The learning-curve diagnostic is OFF by default and renders only when LearningCurveConfig(enabled=True)."""
    skip_if_dependency_missing("hgb")
    df = _make_frame(700, binary=True)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    cfg = ReportingConfig(
        show_perf_chart=False, show_fi=False, plot_outputs="matplotlib[png]",
        # Disable the other standalone diagnostics so this run is fast + isolates the learning-curve artifact.
        pdp_ice=False, slice_finder=False, decision_curve=False, shap_panels=False,
        calibration_drift=False, target_acf=False, model_comparison=False,
        learning_curve=LearningCurveConfig(enabled=True, sizes=(0.3, 0.6, 1.0)),
    )
    data_dir = str(tmp_path)
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="diag_lc",
        model_name="hgb_lc",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        reporting_config=cfg,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models", save_charts=True),
        verbose=0,
    )
    files = _saved_chart_files(data_dir)
    joined = " ".join(files)
    assert "learning_curve" in joined, f"learning curve not saved when opted in; files={files}"


def test_learning_curve_off_by_default(tmp_path, reporting_cfg):
    """Default config has no learning curve -- it is the opt-in cost-gated exception."""
    skip_if_dependency_missing("hgb")
    df = _make_frame(700, binary=True)
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
    data_dir = str(tmp_path)
    train_mlframe_models_suite(
        df=df,
        target_name="diag_lc_off",
        model_name="hgb_lc_off",
        features_and_targets_extractor=fte,
        mlframe_models=["hgb"],
        hyperparams_config=get_cpu_config("hgb", 20),
        reporting_config=reporting_cfg,
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=data_dir, models_dir="models", save_charts=True),
        verbose=0,
    )
    files = _saved_chart_files(data_dir)
    assert not any("learning_curve" in f for f in files), f"learning curve rendered while off by default; files={files}"
