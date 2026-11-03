"""flatten_classification_report keeps top-level scalar entries instead of crashing on them."""

from __future__ import annotations

import pytest

pytest.importorskip("mlflow", reason="mlflow ships as an optional extra, not installed in the base CI env")

from mlframe.integrations.mlflow import flatten_classification_report


def test_scalar_entry_outside_separate_metrics_is_kept():
    """A report holding a float or int at top level (e.g. a custom ``log_loss``) flattens to that scalar."""
    cr = {"0": {"precision": 0.5, "recall": 0.4}, "log_loss": 0.31, "n_rows": 120, "accuracy": 0.8}
    flat = flatten_classification_report(cr, source="val_")
    assert flat["val_log_loss"] == 0.31
    assert flat["val_n_rows"] == 120
    assert flat["val_accuracy"] == 0.8
    assert flat["val_class 0_precision"] == 0.5
