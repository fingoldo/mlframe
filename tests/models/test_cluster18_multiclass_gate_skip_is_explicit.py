"""Cluster 18 regression: a multiclass blend-weight skip says why, instead of blaming the row count (audit 2026-11-20, finding 18-F2)."""

import logging

import numpy as np

from mlframe.models.ensembling.score_flavours import run_stacking_aware_gate


def test_multiclass_members_are_reported_as_unsupported(caplog):
    """(N, K>2) member probabilities produce an explicit multiclass-unsupported warning (18-F2)."""
    n, k = 60, 3
    rng = np.random.default_rng(0)
    y = rng.integers(0, k, size=n)
    probs = [rng.dirichlet(np.ones(k), size=n) for _ in range(2)]
    res: dict = {}
    with caplog.at_level(logging.WARNING):
        run_stacking_aware_gate(
            enable_stacking_aware_gate=True,
            _gate_preds_for_check=probs,
            target_arr=y,
            level_models_and_predictions=[object(), object()],
            _ensemble_member_tags=["a", "b"],
            stacking_gate_min_weight=0.0,
            use_nnls_weights=True,
            res=res,
            verbose=False,
        )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "multiclass" in messages and "not supported" in messages, messages
