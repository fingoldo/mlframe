"""Unit + biz_value coverage for ``core._diversity_recommendations.compute_diversity_recommendations``.

Wires ``votenrank.correlation_diversity_ablation.recommend_diversity_additions`` into the suite's
per-target ensembling step as a default-ON diagnostic (wave 2, batch I). Previously had zero
dedicated test coverage. This function is a thin pure-Python adapter (no model fitting, no GPU) --
tested directly against ``SimpleNamespace``-mocked "ensemble members" carrying the same
``oof_preds``/``oof_probs``/``oof_target``/``metrics``/``model`` attribute shape the real suite
stamps onto its fitted models, per ``2026-07-13``'s OOF-wiring fix
(``src/mlframe/training/core/DEFAULTS_CHANGELOG.md``) that made ``oof_target`` actually reach these
entries for regression targets.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training.core._diversity_recommendations import compute_diversity_recommendations
from mlframe.training.configs import TargetTypes


def _member(name, oof_preds, rmse):
    return SimpleNamespace(
        oof_preds=oof_preds,
        oof_target=None,  # set by caller (shared across members)
        model=SimpleNamespace(),
        model_name=name,
        metrics={"val": {"rmse": rmse}},
    )


def _regression_pool(seed=1, n=4000, n_models=6, diverse_noise=0.6):
    """A pool where half the models are tight/correlated and half are diverse-but-weaker.

    Noise ratio (0.3 tight vs 0.6 diverse -- 2x, not the much wider gap
    ``bench_correlation_diversity_ablation.py``'s own bench uses purely for timing) is tuned so the
    diverse half's blend contribution is measurably positive: too weak a diverse group (much wider
    noise gap) drags the blend down instead of helping it, correctly yielding an empty shortlist --
    confirmed directly against ``recommend_diversity_additions`` before picking this ratio.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=n)
    members = []
    for i in range(n_models):
        noise_scale = 0.3 if i < n_models // 2 else diverse_noise
        pred = y_true + noise_scale * rng.standard_normal(n)
        rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
        m = _member(f"m{i}", pred, rmse)
        m.oof_target = y_true
        members.append(m)
    return members


def test_biz_val_diversity_recommendations_surfaces_diverse_weaker_member():
    """A diverse-but-individually-weaker member must be recommended over a tight, correlated cluster."""
    members = _regression_pool()
    behavior_config = SimpleNamespace(
        recommend_diversity_additions_in_leaderboard=True,
        diversity_recommendation_correlation_threshold=0.85,
        diversity_recommendation_min_improvement=0.0,
        diversity_recommendation_top_k=None,
    )
    shortlist = compute_diversity_recommendations(ens_models=members, target_type=TargetTypes.REGRESSION, behavior_config=behavior_config, verbose=False)
    assert shortlist is not None, "diversity recommendation did not fire on a well-formed OOF pool"
    assert len(shortlist) >= 1, "no diverse-but-weaker member was recommended despite a genuine diversity gap"


def test_default_off_when_flag_false():
    """Explicit opt-out must return None without computing anything."""
    members = _regression_pool()
    behavior_config = SimpleNamespace(recommend_diversity_additions_in_leaderboard=False)
    assert compute_diversity_recommendations(ens_models=members, target_type=TargetTypes.REGRESSION, behavior_config=behavior_config) is None


def test_none_when_fewer_than_two_members():
    members = _regression_pool(n_models=1)
    behavior_config = SimpleNamespace(recommend_diversity_additions_in_leaderboard=True)
    assert compute_diversity_recommendations(ens_models=members, target_type=TargetTypes.REGRESSION, behavior_config=behavior_config) is None


def test_none_when_any_member_missing_oof_target():
    """A single member missing oof_target must silently no-op the whole diagnostic (documented contract)."""
    members = _regression_pool()
    members[0].oof_target = None
    behavior_config = SimpleNamespace(recommend_diversity_additions_in_leaderboard=True)
    assert compute_diversity_recommendations(ens_models=members, target_type=TargetTypes.REGRESSION, behavior_config=behavior_config) is None


def test_none_when_any_member_missing_oof_preds():
    members = _regression_pool()
    members[0].oof_preds = None
    behavior_config = SimpleNamespace(recommend_diversity_additions_in_leaderboard=True)
    assert compute_diversity_recommendations(ens_models=members, target_type=TargetTypes.REGRESSION, behavior_config=behavior_config) is None


def test_none_for_multiclass():
    """Multiclass oof_probs (n, C>=3) has no single-column diversity proxy -- documented skip, not a crash."""
    members = _regression_pool()
    behavior_config = SimpleNamespace(recommend_diversity_additions_in_leaderboard=True)
    assert compute_diversity_recommendations(ens_models=members, target_type=TargetTypes.MULTICLASS_CLASSIFICATION, behavior_config=behavior_config) is None
