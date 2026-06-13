"""biz_value: the production ensemble-flavour chooser ranks by discrimination (roc_auc) first.

Default lives in ``training/core/_ensemble_chooser.py::_CLASSIFICATION_METRICS`` and is REPLAYED
verbatim at predict time (``predict._resolve_chosen_flavour``). bench_ensemble_chooser_rank_metric
shows ranking flavours by OOF ROC-AUC beats calibration(ICE)-first on honest held-out test ROC-AUC
in 21/21 synthetic cells (mean +0.0024). These tests pin that the REAL chooser picks the
higher-AUC flavour when discrimination and calibration disagree, and that the pre-flip ICE-first
order would have picked the worse-discriminating one.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.models.ensembling.base import SIMPLE_ENSEMBLING_METHODS, combine_probs
from mlframe.metrics._ice_metric import compute_probabilistic_multiclass_error
from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour


class _FakeEns:
    def __init__(self, metrics):
        self.metrics = metrics


def _two_col(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.column_stack([1.0 - p, p])


def _build_ensembles_from_members(members, y_oof, oof_i, y_test, test_i):
    """Build {flavour: _FakeEns} carrying real OOF ice/roc_auc + honest test roc_auc."""
    out = {}
    for flav in SIMPLE_ENSEMBLING_METHODS:
        c_oof = combine_probs(members[:, oof_i], flav)
        c_test = combine_probs(members[:, test_i], flav)
        ice = compute_probabilistic_multiclass_error(
            y_true=y_oof, y_score=_two_col(c_oof), method="multicrit", nbins=10,
        )
        out[flav] = (
            _FakeEns({
                "oof": {1: {"ice": float(ice), "roc_auc": float(roc_auc_score(y_oof, c_oof))}},
                "test": {1: {"roc_auc": float(roc_auc_score(y_test, c_test))}},
            }),
            float(roc_auc_score(y_test, c_test)),
        )
    return out


def _mixed_skill_cell(seed=1):
    rng = np.random.default_rng(1000 + seed)
    n = 5000
    y = (rng.random(n) < 0.25).astype(int)
    specs = [(1.4, 0.0, 1.0), (0.4, 0.5, 1.2), (1.0, -0.3, 0.9), (0.7, 0.1, 1.1), (0.3, 0.0, 1.0)]
    members = []
    for skill, bias, temp in specs:
        signal = np.where(y == 1, rng.normal(1.2, 1.0, n), rng.normal(-1.2, 1.0, n))
        z = temp * (signal * skill) + bias + rng.normal(0, 0.3, n)
        members.append(1.0 / (1.0 + np.exp(-z)))
    members = np.stack(members)
    idx = rng.permutation(n)
    cut = n // 2
    return members, y, idx[:cut], idx[cut:]


def test_biz_chooser_picks_higher_test_auc_flavour():
    """REAL chooser's winner must have test ROC-AUC within noise of the BEST available flavour, and
    strictly beat the flavour the pre-flip ICE-first order would have selected."""
    members, y, oof_i, test_i = _mixed_skill_cell(seed=1)
    ens = _build_ensembles_from_members(members, y[oof_i], oof_i, y[test_i], test_i)
    ensembles = {k: v[0] for k, v in ens.items()}
    test_auc = {k: v[1] for k, v in ens.items()}

    winner = _choose_ensemble_flavour(ensembles)
    best_flav = max(test_auc, key=test_auc.get)

    # Production (AUC-first) winner is the best-discriminating flavour on the honest split.
    assert test_auc[winner] >= test_auc[best_flav] - 1e-6, (
        f"chooser picked {winner} (test AUC {test_auc[winner]:.5f}) but best is "
        f"{best_flav} ({test_auc[best_flav]:.5f})"
    )

    # What ICE-first WOULD have picked (lowest OOF ice). It must be measurably worse on test AUC --
    # this is the regression guard: reverting the flip drops honest test AUC here.
    ice_pick = min(ensembles, key=lambda k: ensembles[k].metrics["oof"][1]["ice"])
    assert test_auc[winner] > test_auc[ice_pick] + 1e-3, (
        f"AUC-first winner {winner} ({test_auc[winner]:.5f}) should beat ICE-first pick "
        f"{ice_pick} ({test_auc[ice_pick]:.5f}) by >1e-3"
    )


def test_biz_chooser_majority_win_across_seeds():
    """Across 3 seeds the AUC-first winner's honest test AUC >= the ICE-first pick's in every cell."""
    wins = 0
    for seed in (0, 1, 2):
        members, y, oof_i, test_i = _mixed_skill_cell(seed=seed)
        ens = _build_ensembles_from_members(members, y[oof_i], oof_i, y[test_i], test_i)
        ensembles = {k: v[0] for k, v in ens.items()}
        test_auc = {k: v[1] for k, v in ens.items()}
        winner = _choose_ensemble_flavour(ensembles)
        ice_pick = min(ensembles, key=lambda k: ensembles[k].metrics["oof"][1]["ice"])
        if test_auc[winner] >= test_auc[ice_pick]:
            wins += 1
    assert wins == 3, f"AUC-first should win/tie ICE-first in all 3 seeds, got {wins}"
