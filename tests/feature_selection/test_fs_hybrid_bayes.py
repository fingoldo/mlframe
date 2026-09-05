"""Behavioural tests for the hierarchical posterior over an arm's pooled advantage.

The quadrature is checked against properties that hold for any correct random-effects fit -- shrinkage
towards the pooled mean, heterogeneity that grows when scenarios disagree, a ROPE curve that is a genuine
CDF -- plus one closed-form anchor: when the scenarios agree exactly, the posterior for the pooled effect
must collapse onto the fixed-effect answer, which is computable by hand.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._bayes import (
    PRIOR_KINDS,
    ScenarioEffect,
    extract_skill_rows,
    fit_hierarchical,
    hierarchical_report,
    prior_sensitivity,
    rope_curve,
    scenario_effects,
)


def _effects(pairs: List[tuple], se: float = 0.01) -> List[ScenarioEffect]:
    """Build effects from ``(scenario, delta)`` pairs at a common standard error."""
    return [ScenarioEffect(scenario=name, m=20, delta=delta, se=se) for name, delta in pairs]


class TestPooledEffect:
    """What the posterior for the pooled advantage must do."""

    def test_needs_at_least_two_scenarios(self) -> None:
        """Heterogeneity is not identified from one scenario, so the fit declines rather than guessing."""
        assert fit_hierarchical(_effects([("a", 0.05)])) is None

    def test_agreeing_scenarios_collapse_onto_the_fixed_effect_answer(self) -> None:
        """When every scenario reports the same delta, tau is near zero and the pooled sd is se/sqrt(k)."""
        effects = _effects([("a", 0.05), ("b", 0.05), ("c", 0.05), ("d", 0.05)], se=0.01)
        fit = fit_hierarchical(effects)
        assert fit is not None
        assert fit.mu_mean == pytest.approx(0.05, abs=0.002)
        assert fit.mu_sd == pytest.approx(0.01 / math.sqrt(4), rel=0.35)
        assert fit.tau_median < 0.01

    def test_pooled_effect_lies_between_the_scenario_effects(self) -> None:
        """Partial pooling cannot land outside the range of what it pools."""
        effects = _effects([("a", 0.02), ("b", 0.06), ("c", 0.10)])
        fit = fit_hierarchical(effects)
        assert fit is not None
        assert 0.02 < fit.mu_mean < 0.10

    def test_a_clear_positive_effect_gets_a_high_probability_of_being_positive(self) -> None:
        """Four consistent gains at ten standard errors leave essentially no posterior mass below zero."""
        fit = fit_hierarchical(_effects([("a", 0.10), ("b", 0.10), ("c", 0.10), ("d", 0.10)], se=0.01))
        assert fit is not None
        assert fit.p_positive > 0.99
        assert fit.p_in_rope < 0.01


class TestHeterogeneity:
    """Tau is the headline quantity, not a nuisance parameter."""

    def test_disagreeing_scenarios_produce_a_larger_tau(self) -> None:
        """Scenarios that disagree far beyond their own error must widen the between-scenario spread."""
        agree = fit_hierarchical(_effects([("a", 0.05), ("b", 0.05), ("c", 0.05)], se=0.01))
        disagree = fit_hierarchical(_effects([("a", -0.10), ("b", 0.05), ("c", 0.20)], se=0.01))
        assert agree is not None and disagree is not None
        assert disagree.tau_median > 10 * agree.tau_median

    def test_heterogeneity_widens_the_pooled_interval(self) -> None:
        """A pooled mean of the same magnitude is less certain when the scenarios disagree about it."""
        agree = fit_hierarchical(_effects([("a", 0.05), ("b", 0.05), ("c", 0.05)], se=0.01))
        disagree = fit_hierarchical(_effects([("a", -0.10), ("b", 0.05), ("c", 0.20)], se=0.01))
        assert agree is not None and disagree is not None
        assert disagree.mu_sd > agree.mu_sd


class TestRopeCurve:
    """The curve is offered instead of arguing about a single radius, so it has to be a real CDF."""

    def test_is_monotone_and_approaches_one(self) -> None:
        """P(|mu| < r) never decreases in r and reaches near-certainty far outside the posterior."""
        effects = _effects([("a", 0.02), ("b", 0.04), ("c", 0.03)])
        curve = rope_curve(effects, radii=[0.001, 0.01, 0.05, 0.2, 1.0])
        masses = [mass for _, mass in curve]
        assert all(b >= a - 1e-12 for a, b in zip(masses, masses[1:]))
        assert masses[-1] > 0.99

    def test_agrees_with_the_fit_at_the_pre_registered_radius(self) -> None:
        """The curve and the fit's own ROPE mass are the same integral, so they must match."""
        effects = _effects([("a", 0.02), ("b", 0.04), ("c", 0.03)])
        fit = fit_hierarchical(effects, rope=0.05)
        assert fit is not None
        ((_, mass),) = rope_curve(effects, radii=[0.05])
        assert mass == pytest.approx(fit.p_in_rope, abs=1e-6)


class TestPriors:
    """The tau prior is a choice, so its influence is reported rather than hidden."""

    def test_every_declared_prior_produces_a_fit(self) -> None:
        """All three pre-registered priors run, so the sensitivity table can never be silently partial."""
        fits = prior_sensitivity(_effects([("a", 0.02), ("b", 0.06), ("c", 0.10)]))
        assert set(fits) == set(PRIOR_KINDS)
        assert all(fit is not None for fit in fits.values())

    def test_a_wider_prior_admits_more_heterogeneity(self) -> None:
        """The wide half-Cauchy cannot report less spread than the half-normal on disagreeing scenarios."""
        effects = _effects([("a", -0.10), ("b", 0.05), ("c", 0.20)], se=0.01)
        fits = prior_sensitivity(effects)
        narrow, wide = fits["half_normal"], fits["half_cauchy_wide"]
        assert narrow is not None and wide is not None
        assert wide.tau_median >= narrow.tau_median

    def test_an_unknown_prior_is_rejected(self) -> None:
        """A typo in the prior name fails loudly instead of silently falling back to a default."""
        with pytest.raises(ValueError, match="unknown tau prior"):
            fit_hierarchical(_effects([("a", 0.02), ("b", 0.06)]), prior="inverse_gamma")


class TestFromRecords:
    """Estimation off cell records, on the normalized-skill scale the ROPE is defined on."""

    @staticmethod
    def _record(scenario: str, arm: str, seed: int, skill: float) -> Dict[str, Any]:
        """Build a minimal ok cell record carrying one lightgbm skill value at k10."""
        return {
            "status": "ok",
            "arm": arm,
            "scenario": scenario,
            "dataset_seed": seed,
            "cv_seed": 0,
            "scores": {"k10": {"skill": {"lightgbm": skill}}},
        }

    def _records(self) -> List[Dict[str, Any]]:
        """Two scenarios where the arm beats the null by a consistent margin on five seeds each."""
        out: List[Dict[str, Any]] = []
        for scenario, gain in (("bed_a", 0.05), ("bed_b", 0.09)):
            for seed in range(5):
                out.append(self._record(scenario, "all-features", seed, 0.30 + 0.01 * seed))
                out.append(self._record(scenario, "arm", seed, 0.30 + 0.01 * seed + gain))
        return out

    def test_reads_skill_not_a_raw_metric(self) -> None:
        """The extractor takes the skill block; a record without one contributes nothing."""
        rows = extract_skill_rows(self._records(), model="lightgbm", k_label="k10")
        assert len(rows) == 20
        assert extract_skill_rows(self._records(), model="logistic", k_label="k10") == []

    def test_effects_are_paired_within_scenario(self) -> None:
        """Each scenario contributes one effect, and it is the mean paired gain, not the level."""
        rows = extract_skill_rows(self._records(), model="lightgbm", k_label="k10")
        effects = {e.scenario: e for e in scenario_effects(rows, arm="arm")}
        assert set(effects) == {"bed_a", "bed_b"}
        assert effects["bed_a"].delta == pytest.approx(0.05)
        assert effects["bed_b"].delta == pytest.approx(0.09)

    def test_report_carries_both_the_pooled_row_and_the_curve(self) -> None:
        """The rendered block states the pooled effect, the heterogeneity and the ROPE curve together."""
        text = hierarchical_report(self._records(), model="lightgbm", k_label="k10")
        assert "P(in ROPE)" in text
        assert "tau (median)" in text
        assert "P(|mu| < r)" in text
