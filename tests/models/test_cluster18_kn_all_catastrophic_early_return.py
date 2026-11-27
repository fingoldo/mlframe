"""Cluster 18 regression: a K>2 gate that leaves one member short-circuits like K=2 instead of blending a one-member ensemble (audit 2026-11-20, 18-F14)."""

from types import SimpleNamespace

import numpy as np

from mlframe.models.ensembling import score_gate


def _run_kn(preds, tags):
    """Run the K>2 absolute-MAE gate on the given member predictions against a known target."""
    n = len(preds[0])
    target = np.linspace(0.0, 1.0, n)
    res: dict = {}
    members = [SimpleNamespace() for _ in preds]
    score_gate.catastrophic_drop_kn(
        level_models_and_predictions=members, _gate_preds_for_check=preds, _gate_source_split="val",
        _ensemble_member_tags=list(tags), _ensemble_short_tags=list(tags),
        train_target_arr=None, val_target_arr=target, test_target_arr=None,
        k2_catastrophic_mae_ratio=5.0, verbose=False, res=res,
    )
    return res, target


def test_one_survivor_stamps_the_k2_style_reason_and_returns_early():
    """Two catastrophic members out of three: the sentinel names the survivor, the helper stamps _reason and asks for the early return (18-F14)."""
    n = 50
    res, _target = _run_kn([target_like(n, 0.01), target_like(n, 50.0), target_like(n, -80.0)], ["good", "bad1", "bad2"])
    assert res["_all_members_catastrophic"]["n_survivors"] == 1
    assert score_gate.kn_all_catastrophic_early_return(res) is True
    assert res["_reason"] == "kn_catastrophic_dropout" and res["_n_members"] == 1
    assert res["_kept_member"] == "good" and res["_dropped_member"] == "bad1, bad2"


def test_healthy_or_partially_dropped_ensembles_do_not_short_circuit():
    """All members fine, or two survivors out of three, keep the normal path (18-F14)."""
    n = 50
    res, _ = _run_kn([target_like(n, 0.01), target_like(n, 0.02), target_like(n, 0.03)], ["a", "b", "c"])
    assert score_gate.kn_all_catastrophic_early_return(res) is False and "_reason" not in res
    res2, _ = _run_kn([target_like(n, 0.01), target_like(n, 0.02), target_like(n, 90.0)], ["a", "b", "c"])
    assert score_gate.kn_all_catastrophic_early_return(res2) is False


def target_like(n, offset):
    """A prediction vector equal to the target plus a constant offset."""
    return np.linspace(0.0, 1.0, n) + offset
