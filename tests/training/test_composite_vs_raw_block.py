"""The composite-vs-raw verdict table lists trained composites only; budget-dropped specs are not "NO_Y_SCALE_METRIC" rows."""

from __future__ import annotations

from types import SimpleNamespace

from mlframe.training.core._phase_composite_post_summary import format_composite_vs_raw_block


def test_untrained_composite_is_not_listed():
    meta = {
        "dummy_baselines": {"regression": {"y": {"primary_metric": "val_RMSE", "strongest": "mean", "data": {"mean": {"val_RMSE": 2.0}}}}},
        "composite_target_y_scale_metrics": {"regression": {"y-logY": [{"model_name": "cb", "metrics": {"val": {"RMSE": 1.5}, "test": {"RMSE": 1.6}}}]}},
    }
    models = {"regression": {"y-logY": [SimpleNamespace(model_name="cb")]}}
    text = format_composite_vs_raw_block(
        models=models, metadata=meta, best_metrics={("regression", "y"): {"val_RMSE": 1.4, "model_name": "raw"}},
        composite_to_raw={("regression", "y-logY"): "y", ("regression", "y-cbrtY"): "y"},
    )
    assert "y-logY" in text
    assert "y-cbrtY" not in text
