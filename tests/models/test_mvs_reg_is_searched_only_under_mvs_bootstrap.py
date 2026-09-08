"""`mvs_reg` was commented out of the CatBoost search space while `MVS` stayed in `bootstrap_type`.

CatBoost accepts `mvs_reg` only when `bootstrap_type` is `MVS` — it balances importance against
Bernoulli sampling in the MVS denominator — and passing it under any other bootstrap raises. The
parameter had been commented out entirely rather than gated, so the tuner explored MVS bootstrapping
with its one tuning knob pinned at CatBoost's default for every trial.

It is enabled here through the mechanism the file already uses for exactly this shape:
`drop_if_not_rules` deletes a field whenever its condition does NOT hold, which is how
`bagging_temperature` is kept to Bayesian bootstrap only. The failure mode of getting this wrong is
loud rather than silent — CatBoost raises on the offending trial — but it would raise on most trials,
so both directions are pinned below.

Written to py-ci-shared/WRITING_TESTS.md habits 1, 2 and 5.
"""

from __future__ import annotations

import pytest

from mlframe.models.tuning_rules import check_rules


def _rules():
    """The CatBoost tuner's own rule set, read off a constructed tuner rather than restated here.

    Restating the rules in the test would let the file drift from the config it is about: the test
    would keep passing against a rule the tuner no longer has.
    """
    from mlframe.models.tuning_catboost import CatboostParamsOptimizer

    tuner = CatboostParamsOptimizer()
    return tuner.drop_if_rules, tuner.drop_if_not_rules


def _survives(params: dict) -> dict:
    """*params* after the tuner's drop rules have run, which is what actually reaches CatBoost."""
    drop_if_rules, drop_if_not_rules = _rules()
    kept = dict(params)
    check_rules(kept, drop_if_rules=drop_if_rules, drop_if_not_rules=drop_if_not_rules)
    return kept


class TestMvsRegFollowsTheBootstrapType:
    """`mvs_reg` is valid under MVS and rejected by CatBoost under every other bootstrap."""

    def test_it_is_kept_under_mvs(self):
        """The point of enabling it. Dropped here too, the parameter would be in the search space
        and never actually reach CatBoost — searched in name only."""
        kept = _survives({"bootstrap_type": "MVS", "mvs_reg": 0.5})

        assert kept.get("mvs_reg") == 0.5

    @pytest.mark.parametrize("bootstrap", ["Bayesian", "Bernoulli", "No", "Poisson"])
    def test_it_is_dropped_under_every_other_bootstrap(self, bootstrap):
        """CatBoost raises when `mvs_reg` arrives under a non-MVS bootstrap, and `bootstrap_type`
        is itself part of the search space — so without this rule most sampled trials would die."""
        kept = _survives({"bootstrap_type": bootstrap, "mvs_reg": 0.5})

        assert "mvs_reg" not in kept

    def test_a_candidate_without_mvs_reg_is_untouched(self):
        """The rule deletes a field; it must not invent one or reject the candidate."""
        kept = _survives({"bootstrap_type": "Bernoulli", "subsample": 0.8})

        assert kept == {"bootstrap_type": "Bernoulli", "subsample": 0.8}


class TestTheSearchSpaceAndTheRuleAgree:
    """The gate and the space have to stay consistent, or one of them is describing nothing."""

    def test_mvs_is_actually_reachable_in_the_bootstrap_search_space(self):
        """The gate is only worth having while MVS is sampled at all. If MVS were removed from the
        space, `mvs_reg` would be dead weight rather than a tunable, and this test says so."""
        from mlframe.models.tuning_catboost import CatboostParamsOptimizer

        assert "MVS" in CatboostParamsOptimizer().params["bootstrap_type"]

    def test_mvs_reg_is_in_the_search_space(self):
        """It spent time commented out while MVS stayed sampled, which is the state this pins shut."""
        from mlframe.models.tuning_catboost import CatboostParamsOptimizer

        assert "mvs_reg" in CatboostParamsOptimizer().params

    def test_the_none_option_is_offered_alongside_the_distribution(self):
        """`None` leaves CatBoost's own default in place, so the tuner can decline to set it at all
        rather than being forced to pick from the distribution on every MVS trial."""
        from mlframe.models.tuning_catboost import CatboostParamsOptimizer

        assert None in CatboostParamsOptimizer().params["mvs_reg"]
