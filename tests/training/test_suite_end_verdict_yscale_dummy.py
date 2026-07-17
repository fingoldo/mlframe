"""Suite-end verdict block must compare a composite model's y-scale RMSE
against a y-scale dummy, never the T-scale (residual) dummy.

``_phase_dummy_baselines`` stamps ``y_scale_strongest_metrics`` (the strongest
dummy inverted to y-scale) precisely so the verdict is apples-to-apples. The
verdict formatter previously ignored it: when no raw-y trivial baseline was
mappable it left ``dummy_val`` on the T-scale and printed it next to the
y-scale model metric -- a scale mismatch the operator reads as a real lift.
"""

from __future__ import annotations

import re

from mlframe.training.baselines import format_suite_end_summary


def _rep(t_scale_rmse: float, y_scale_rmse: float) -> dict:
    """A composite-target dummy report: strongest dummy on T-scale + its
    y-scale inversion under ``y_scale_strongest_metrics``."""
    return {
        "strongest": "median",
        "primary_metric": "val_RMSE",
        "data": {"median": {"val_RMSE": t_scale_rmse}},
        "y_scale_strongest_metrics": {"val": {"RMSE": y_scale_rmse, "MAE": 0.0}},
    }


def test_verdict_uses_yscale_dummy_when_no_raw_map():
    # T-scale residual RMSE ~ 1.2 (tiny); y-scale dummy RMSE ~ 480 (original units).
    db = {"regression": {"y-linres-base": _rep(t_scale_rmse=1.2, y_scale_rmse=480.0)}}
    # Model is on the y-scale (RMSE 300, beats the y-scale dummy 480 -> lift 1.6x).
    best = {("regression", "y-linres-base"): {"val_RMSE": 300.0, "model_name": "Composite"}}
    out = format_suite_end_summary(db, best_model_metrics_by_target=best, min_lift=1.5)

    m = re.search(r"val_RMSE=([0-9.]+)\s+Composite", out)
    assert m, f"composite row with dummy_metric not found:\n{out}"
    dummy_printed = float(m.group(1))
    # Pre-fix bug: dummy_printed == 1.2 (T-scale), yielding lift 0.0040x and a
    # false MODELS_BARELY_BEAT_TRIVIAL. Post-fix: dummy_printed == 480 (y-scale).
    assert abs(dummy_printed - 480.0) < 1e-6, f"dummy_metric printed on the wrong scale: {dummy_printed} (expected 480 y-scale)"
    assert "y-scale inv" in out
    assert "TASK_NON_TRIVIAL_AND_MODELS_HEALTHY" in out
    assert "BEST_MODEL_BELOW_DUMMY" not in out


def test_verdict_raw_y_map_still_preferred_over_yscale_inv():
    # When a raw-y trivial baseline IS mappable, it remains the preferred dummy
    # (truly-trivial median(y_raw)), not the fitted-alpha y-scale inversion.
    db = {
        "regression": {
            "y-linres-base": _rep(t_scale_rmse=1.2, y_scale_rmse=480.0),
            "y": {
                "strongest": "median",
                "primary_metric": "val_RMSE",
                "data": {"median": {"val_RMSE": 510.0}},
            },
        }
    }
    best = {("regression", "y-linres-base"): {"val_RMSE": 300.0, "model_name": "Composite"}}
    cmap = {("regression", "y-linres-base"): "y"}
    out = format_suite_end_summary(db, best_model_metrics_by_target=best, composite_to_raw_target_map=cmap, min_lift=1.5)
    m = re.search(r"val_RMSE=([0-9.]+)\s+Composite", out)
    assert m, f"composite row not found:\n{out}"
    assert abs(float(m.group(1)) - 510.0) < 1e-6, "raw-y trivial dummy must win over y-scale inv"
    assert "raw-y trivial" in out


def test_composite_without_yscale_model_metric_does_not_fall_through_to_t_scale(caplog):
    """A composite whose y-scale MODEL metric is missing must NOT borrow its T-scale (residual) model
    metric for the verdict -- that mixed a ~1.2 residual RMSE against a ~13 y-scale dummy and printed a
    false 9x lift / TASK_NON_TRIVIAL_AND_MODELS_HEALTHY while the model's real y-scale R^2 was -146.
    With no usable y-scale model metric the composite row must honestly show best_model='-'.
    """
    import logging
    from types import SimpleNamespace

    from mlframe.training.configs import DummyBaselinesConfig
    from mlframe.training.core._phase_composite_post_summary import _run_suite_end_dummy_baselines_summary

    # Model entry carries only T-scale (residual) metrics ~1.2.
    entry = SimpleNamespace(metrics={"val": {"RMSE": 1.2}, "test": {"RMSE": 1.2}}, model_name="Composite")
    models = {"regression": {"TVT-linres-base": [entry]}}
    metadata = {
        "dummy_baselines": {
            "regression": {
                "TVT-linres-base": {
                    "strongest": "median",
                    "primary_metric": "val_RMSE",
                    "data": {"median": {"val_RMSE": 13.0}},
                    "y_scale_strongest_metrics": {"val": {"RMSE": 13.0}, "MAE": 0.0},
                }
            }
        },
        # Composite target: KEY PRESENT but no usable y-scale model metric (empty list).
        "composite_target_y_scale_metrics": {"regression": {"TVT-linres-base": []}},
    }
    with caplog.at_level(logging.INFO):
        _run_suite_end_dummy_baselines_summary(
            models=models,
            metadata=metadata,
            dummy_baselines_config=DummyBaselinesConfig(),
        )
    text = caplog.text
    assert "TVT-linres-base" in text, f"composite row missing from verdict:\n{text}"
    # Must NOT manufacture a TASK_HEALTHY verdict from the T-scale residual metric.
    assert "TASK_NON_TRIVIAL_AND_MODELS_HEALTHY" not in text, f"false TASK_HEALTHY from T-scale fallthrough:\n{text}"
