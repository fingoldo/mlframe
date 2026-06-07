"""Regression: CompositeTargetDiscoveryConfig defaults must not silently
fit an in-sample-stacked ensemble.

Pre-fix: ``cross_target_ensemble_strategy="nnls_stack"`` (stacker) combined
with ``oof_holdout_frac=0.0`` meant NNLS fit on TRAIN component predictions
where each component had effectively memorised its training rows. Classic
stacking leak.
"""

from __future__ import annotations

from mlframe.training.configs import CompositeTargetDiscoveryConfig


def test_default_does_not_fit_in_sample_stacker():
    cfg = CompositeTargetDiscoveryConfig()
    # Acceptable defenses: either the strategy isn't a stacker, or the
    # stacker fits on a real honest holdout slice (>0).
    is_stacker = cfg.cross_target_ensemble_strategy in {"linear_stack", "nnls_stack"}
    if is_stacker:
        assert cfg.oof_holdout_frac > 0.0, (
            "Default cross_target_ensemble_strategy is a stacker "
            f"({cfg.cross_target_ensemble_strategy!r}) but oof_holdout_frac=0.0; "
            "this fits the stacker on in-sample component predictions (leak)."
        )
    else:
        # Non-stacker default is also acceptable.
        assert cfg.cross_target_ensemble_strategy in {"off", "mean", "oof_weighted"}


def test_oof_holdout_frac_default_value():
    # Explicit pin so the safe-default doesn't regress to 0.0 silently.
    cfg = CompositeTargetDiscoveryConfig()
    assert 0.0 < cfg.oof_holdout_frac < 1.0
