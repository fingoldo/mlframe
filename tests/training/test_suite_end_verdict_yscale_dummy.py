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
    assert abs(dummy_printed - 480.0) < 1e-6, (
        f"dummy_metric printed on the wrong scale: {dummy_printed} (expected 480 y-scale)"
    )
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
    out = format_suite_end_summary(
        db, best_model_metrics_by_target=best, composite_to_raw_target_map=cmap, min_lift=1.5
    )
    m = re.search(r"val_RMSE=([0-9.]+)\s+Composite", out)
    assert m, f"composite row not found:\n{out}"
    assert abs(float(m.group(1)) - 510.0) < 1e-6, "raw-y trivial dummy must win over y-scale inv"
    assert "raw-y trivial" in out
