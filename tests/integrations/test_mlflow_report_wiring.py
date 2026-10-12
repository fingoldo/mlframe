"""The evaluation report path logs its classification report to an MLflow run the caller already opened."""

from __future__ import annotations

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow", reason="mlflow ships as an optional extra, not installed in the base CI env")

from mlframe.evaluation.reports import _evaluate_estimator_show_classification_report


def _report_block(results_log: dict) -> tuple:
    """Run the production classification-report block on a fixed binary prediction set."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    preds = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    return _evaluate_estimator_show_classification_report(True, y_true, preds, 2, ["neg", "pos"], results_log, None, None)


def test_classification_report_reaches_active_mlflow_run(tmp_path):
    """With a real MLflow run open, the per-class, macro and accuracy metrics land in that run's metric history."""
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("report_wiring")
    with mlflow.start_run() as run:
        report_dict, _report_text = _report_block({"results": {}})
        run_id = run.info.run_id

    logged = mlflow.MlflowClient().get_run(run_id).data.metrics
    assert report_dict["accuracy"] == pytest.approx(0.75)
    assert logged["accuracy"] == pytest.approx(0.75)
    assert logged["balanced_accuracy"] == pytest.approx(report_dict["balanced_accuracy"])
    assert logged["class pos_precision"] == pytest.approx(report_dict["pos"]["precision"])
    assert logged["class neg_recall"] == pytest.approx(report_dict["neg"]["recall"])
    assert logged["macro avg_f1-score"] == pytest.approx(report_dict["macro avg"]["f1-score"])


def test_classification_report_without_active_run_logs_nothing(tmp_path):
    """Outside an MLflow run the report block returns the same dict and creates no run."""
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow_idle.db').as_posix()}")
    mlflow.set_experiment("report_wiring_idle")
    assert mlflow.active_run() is None

    report_dict, _report_text = _report_block({"results": {}})

    assert report_dict["accuracy"] == pytest.approx(0.75)
    assert mlflow.search_runs(experiment_names=["report_wiring_idle"], output_format="list") == []
