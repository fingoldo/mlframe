"""Fitted blend weights are applied only when the gate predictions are out-of-fold or calibration-slice predictions."""

from __future__ import annotations

import types

import numpy as np
import pytest

from mlframe.models.ensembling.score_flavours import run_stacking_aware_gate


def _run(source):
    """Run the gate on a strong and a weak member with the given gate source split."""
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.4).astype(np.float64)
    preds = [np.clip(y * 0.8 + rng.normal(0, 0.1, 300) + 0.1, 0, 1), rng.random(300)]
    res: dict = {}
    weights = run_stacking_aware_gate(
        enable_stacking_aware_gate=True,
        _gate_preds_for_check=preds,
        target_arr=y,
        level_models_and_predictions=[types.SimpleNamespace(), types.SimpleNamespace()],
        _ensemble_member_tags=["good", "weak"],
        stacking_gate_min_weight=0.01,
        use_nnls_weights=True,
        res=res,
        verbose=False,
        gate_source_split=source,
    )
    return weights, res


@pytest.mark.parametrize("source", ["val", "val-coarse", "train", "test"])
def test_weights_not_applied_on_early_stopped_split(source):
    """Weights fitted on a split that drove early stopping stay out of the blend."""
    weights, res = _run(source)
    assert weights is None
    assert res["_stacking_gate"]["applied_to_blend"] is False
    assert "not out-of-fold" in res["_stacking_gate"]["weights_skipped_reason"]


@pytest.mark.parametrize("source", ["oof", "calib", None])
def test_weights_applied_on_out_of_fold_source(source):
    """OOF and calibration-slice predictions still yield blend weights."""
    weights, res = _run(source)
    assert weights is not None and abs(float(weights.sum()) - 1.0) < 1e-9
    assert res["_stacking_gate"]["applied_to_blend"] is True
