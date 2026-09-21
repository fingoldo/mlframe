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


def _decide(model, target, *, skip_models=_SKIP_SET, enabled=True, lag1=1.0, group_aware=True, threshold=0.99):
    """Decide."""
    return extreme_ar_skip_decision(
        model,
        target,
        skip_models=skip_models,
        skip_enabled=enabled,
        lag1_autocorr_per_group=lag1,
        group_aware=group_aware,
        threshold=threshold,
    )


def test_raw_target_neural_family_is_skipped() -> None:
    """Raw target neural family is skipped."""
    for m in _NEURAL:
        skip, fired = _decide(m, "TVT")
        assert skip is True, f"{m} should be skipped on raw extreme-AR target"
        assert fired is True


def test_composite_target_never_skipped_for_any_neural() -> None:
    # Composite targets bound the variance -> neural nets MUST train there.
    """Composite target never skipped for any neural."""
    for m in _NEURAL:
        for comp in ("TVT-diff-kf_tvt_post_mean", "TVT-linresR-TVT_prev", "TVT-poly2-TVT_prev", "TVT-addres-TVT_prev"):
            skip, fired = _decide(m, comp)
            assert skip is False, f"{m} must NOT be skipped on composite {comp}"
            assert fired is False, "AR signal does not apply to composite target"


def test_trees_and_linear_not_skipped() -> None:
    """Trees and linear not skipped."""
    for m in ("cb", "xgb", "lgb", "hgb", "linear", "ridge", "lasso"):
        skip, _fired = _decide(m, "TVT")
        assert skip is False, f"{m} must not be gated by default"


def test_mlp_fired_flag_set_even_when_skip_disabled() -> None:
    # Hard skip off: MLP still trains, but `fired` drives its protections.
    """Mlp fired flag set even when skip disabled."""
    skip, fired = _decide("mlp", "TVT", enabled=False)
    assert skip is False
    assert fired is True


def test_no_fire_without_group_aware_split() -> None:
    """No fire without group aware split."""
    skip, fired = _decide("mlp", "TVT", group_aware=False)
    assert skip is False and fired is False


def test_no_fire_below_lag1_threshold() -> None:
    """No fire below lag1 threshold."""
    skip, fired = _decide("mlp", "TVT", lag1=0.80)
    assert skip is False and fired is False
    # at/above threshold fires
    skip2, fired2 = _decide("mlp", "TVT", lag1=0.99)
    assert skip2 is True and fired2 is True


def test_missing_lag1_does_not_fire() -> None:
    """Missing lag1 does not fire."""
    skip, fired = _decide("mlp", "TVT", lag1=None)
    assert skip is False and fired is False


class TestDiscoverySkipNotFiredLogLevel:
    """The discovery-level "skip did NOT fire" dump is a WARNING only when the skip was applicable but lacked data."""

    @staticmethod
    def _blocked(**kw):
        """Call the missing-info blocker with grouped, bounded-zoo defaults overridden by kw."""
        from mlframe.training.core._ar_skip import _extreme_ar_skip_blocked_by_missing_info

        args = dict(group_aware_active=True, bounded_only_zoo=True, lag1_ar=None, is_picked_target=True, threshold=0.99)
        args.update(kw)
        return _extreme_ar_skip_blocked_by_missing_info(**args)

    def test_non_grouped_run_is_not_a_warning(self):
        """No groups: the skip cannot fire by design and per-group lag1 is legitimately absent."""
        assert self._blocked(group_aware_active=False) is False

    def test_unbounded_zoo_is_not_a_warning(self):
        """A zoo with unbounded models does not warn about the skip being blocked."""
        assert self._blocked(bounded_only_zoo=False) is False

    def test_missing_lag1_on_grouped_run_warns(self):
        """A grouped run with no measured lag-1 autocorrelation warns that the skip is blocked."""
        assert self._blocked(lag1_ar=None) is True

    def test_measured_low_lag1_is_not_a_warning(self):
        """A measured low lag-1 autocorrelation is a legitimate reason not to skip, not a warning."""
        assert self._blocked(lag1_ar=0.3) is False

    def test_high_lag1_on_another_target_warns(self):
        """High lag-1 on a target other than the picked one still warns."""
        assert self._blocked(lag1_ar=0.999, is_picked_target=False) is True
