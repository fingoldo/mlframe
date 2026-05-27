"""Unit tests for the extreme-AR + group-aware model-skip decision.

The skip is a COST optimisation: on a group-aware split + lag1~1.0 RAW
target, neural nets collapse (and are dropped by the ensemble gate anyway),
so skip the expensive fit. Critically it must:
  * fire ONLY on the RAW target, NEVER on a composite (diff/linres/residual
    targets bound the variance -- neural nets belong there);
  * gate the whole neural family (mlp/ngb/lstm/gru/rnn/transformer);
  * NEVER gate trees (cb/xgb/lgb) or, by default, the linear family;
  * still report ``fired`` for the MLP so its weight_decay / output-act
    protections engage even when the hard skip is disabled.
"""
from __future__ import annotations

from mlframe.training.core._phase_train_one_target_mlp_helpers import (
    extreme_ar_skip_decision,
)

_NEURAL = ("mlp", "ngb", "lstm", "gru", "rnn", "transformer")
_SKIP_SET = ("mlp", "ngb", "lstm", "gru", "rnn", "transformer")


def _decide(model, target, *, skip_models=_SKIP_SET, enabled=True,
            lag1=1.0, group_aware=True, threshold=0.99):
    return extreme_ar_skip_decision(
        model, target,
        skip_models=skip_models, skip_enabled=enabled,
        lag1_autocorr_per_group=lag1, group_aware=group_aware,
        threshold=threshold,
    )


def test_raw_target_neural_family_is_skipped() -> None:
    for m in _NEURAL:
        skip, fired = _decide(m, "TVT")
        assert skip is True, f"{m} should be skipped on raw extreme-AR target"
        assert fired is True


def test_composite_target_never_skipped_for_any_neural() -> None:
    # Composite targets bound the variance -> neural nets MUST train there.
    for m in _NEURAL:
        for comp in ("TVT-diff-kf_tvt_post_mean", "TVT-linresR-TVT_prev",
                     "TVT-poly2-TVT_prev", "TVT-addres-TVT_prev"):
            skip, fired = _decide(m, comp)
            assert skip is False, f"{m} must NOT be skipped on composite {comp}"
            assert fired is False, "AR signal does not apply to composite target"


def test_trees_and_linear_not_skipped() -> None:
    for m in ("cb", "xgb", "lgb", "hgb", "linear", "ridge", "lasso"):
        skip, fired = _decide(m, "TVT")
        assert skip is False, f"{m} must not be gated by default"


def test_mlp_fired_flag_set_even_when_skip_disabled() -> None:
    # Hard skip off: MLP still trains, but `fired` drives its protections.
    skip, fired = _decide("mlp", "TVT", enabled=False)
    assert skip is False
    assert fired is True


def test_no_fire_without_group_aware_split() -> None:
    skip, fired = _decide("mlp", "TVT", group_aware=False)
    assert skip is False and fired is False


def test_no_fire_below_lag1_threshold() -> None:
    skip, fired = _decide("mlp", "TVT", lag1=0.80)
    assert skip is False and fired is False
    # at/above threshold fires
    skip2, fired2 = _decide("mlp", "TVT", lag1=0.99)
    assert skip2 is True and fired2 is True


def test_missing_lag1_does_not_fire() -> None:
    skip, fired = _decide("mlp", "TVT", lag1=None)
    assert skip is False and fired is False
