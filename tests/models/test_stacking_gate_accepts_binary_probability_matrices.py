"""Blend-weight fitting must see binary (N, 2) probability matrices, not skip every member for a length mismatch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.models.ensembling.score_flavours import run_stacking_aware_gate


def test_two_column_probabilities_get_blend_weights():
    """Two (N, 2) members, one informative and one noise: weights are fitted (not skipped) and favour the informative one."""
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.5).astype(float)
    good = np.clip(0.5 + 0.4 * (y - 0.5) + rng.normal(scale=0.1, size=n), 0.01, 0.99)
    noise = rng.uniform(0.01, 0.99, size=n)
    members = [np.column_stack([1 - p, p]) for p in (good, noise)]
    res: dict = {}
    weights = run_stacking_aware_gate(
        enable_stacking_aware_gate=True,
        _gate_preds_for_check=members,
        target_arr=y,
        level_models_and_predictions=[SimpleNamespace(model_name="a"), SimpleNamespace(model_name="b")],
        _ensemble_member_tags=["a", "b"],
        stacking_gate_min_weight=0.0,
        use_nnls_weights=True,
        res=res,
        verbose=False,
    )
    assert weights is not None and weights[0] > weights[1]
