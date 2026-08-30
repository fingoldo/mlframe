"""Every crisp classification metric the suite prints describes a decision rule at 0.5 that nobody chose.

At the 2.6% base rate of the run that motivated this, 0.5 is close to "predict nobody", so the printed
precision/recall pair described a rule no operator would deploy. The block reports the rule the OOF data
actually supports next to the one that was used -- and deliberately does not apply it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._honest_decision_threshold import (
    MIN_POSITIVES_FOR_A_RECOMMENDATION,
    decision_threshold_block,
    format_decision_threshold_line,
)


@pytest.fixture(scope="module")
def rare_positive_oof():
    """A 2.6%-positive OOF surface with real but imperfect separation, like the production run."""
    rng = np.random.default_rng(7)
    n = 20_000
    y = (rng.random(n) < 0.026).astype(float)
    p = np.clip(rng.normal(0.03 + 0.25 * y, 0.05), 0.0, 1.0)
    return y, p


class TestTheSelectionSurface:
    """Where the threshold may and may not be fitted."""

    def test_it_reports_which_split_it_fitted_on(self, rare_positive_oof):
        """The block has to say where the threshold came from; a number without a split is not auditable."""
        y, p = rare_positive_oof
        assert decision_threshold_block(None, oof_probs=p, oof_target=y)["fitted_on"] == "oof"

    def test_no_oof_is_a_skip_not_a_fallback_to_val(self):
        """Falling back to val would return a number that looks honest and is not -- val drove early stopping."""
        out = decision_threshold_block(None, oof_probs=None, oof_target=None)
        assert out["status"] == "skipped"
        assert "val" in out["reason"] and "test" in out["reason"]

    def test_a_non_binary_target_is_skipped(self):
        """A multiclass target has no single decision threshold to tune."""
        y = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 2.0])
        assert decision_threshold_block(None, oof_probs=np.linspace(0, 1, 6), oof_target=y)["status"] == "skipped"


class TestTheObjective:
    """What is being optimised, and whether the caller said so."""

    def test_costs_select_the_cost_objective(self, rare_positive_oof):
        """Given a cost ratio, average cost per row is the thing to minimise."""
        y, p = rare_positive_oof
        out = decision_threshold_block(None, oof_probs=p, oof_target=y, decision_costs={"fp": 1.0, "fn": 12.0})
        assert out["objective"] == "cost"

    def test_no_costs_falls_back_to_f1_and_says_so(self, rare_positive_oof):
        """F1 prices a false positive and a false negative equally, so the fallback cannot be a recommendation."""
        y, p = rare_positive_oof
        out = decision_threshold_block(None, oof_probs=p, oof_target=y)
        assert out["objective"] == "f1"
        assert out["recommended"] is False, "F1 prices FP and FN equally, so it cannot carry a recommendation"
        assert "F1" in format_decision_threshold_line("m", out)

    def test_an_expensive_false_negative_lowers_the_threshold(self, rare_positive_oof):
        """The whole point of a cost ratio: missing a positive costs 12x, so catch more of them."""
        y, p = rare_positive_oof
        cheap = decision_threshold_block(None, oof_probs=p, oof_target=y, decision_costs={"fp": 1.0, "fn": 1.0})
        dear = decision_threshold_block(None, oof_probs=p, oof_target=y, decision_costs={"fp": 1.0, "fn": 30.0})
        assert dear["tuned"]["threshold"] < cheap["tuned"]["threshold"]
        assert dear["tuned"]["recall"] > cheap["tuned"]["recall"]

    def test_the_tuned_rule_really_costs_less_than_0_5(self, rare_positive_oof):
        """A tuned threshold that does not beat the default on its own objective is a bug, not a nuance."""
        y, p = rare_positive_oof
        out = decision_threshold_block(None, oof_probs=p, oof_target=y, decision_costs={"fp": 1.0, "fn": 12.0})
        assert out["tuned"]["avg_cost"] <= out["default"]["avg_cost"]

    def test_0_5_is_reported_as_the_rule_actually_in_force(self, rare_positive_oof):
        """The default rule must be reported too, because it is the one every other metric used."""
        y, p = rare_positive_oof
        out = decision_threshold_block(None, oof_probs=p, oof_target=y)
        assert out["default"]["threshold"] == 0.5
        assert out["applied"] is False


class TestTheInterval:
    """A threshold is a fitted parameter; its spread has to be visible."""

    def test_a_ci_is_returned_and_brackets_the_estimate(self, rare_positive_oof):
        """A degenerate interval would be worse than none; lo must not exceed hi."""
        y, p = rare_positive_oof
        out = decision_threshold_block(None, oof_probs=p, oof_target=y, decision_costs={"fp": 1.0, "fn": 12.0}, n_boot=100)
        lo, hi = out["threshold_ci"]
        assert lo <= hi

    def test_too_few_positives_is_never_a_recommendation(self):
        """With a handful of positives the optimum is whichever ones landed in the fold."""
        rng = np.random.default_rng(3)
        y = np.zeros(400)
        y[:10] = 1.0
        p = np.clip(rng.normal(0.1 + 0.3 * y, 0.1), 0, 1)
        out = decision_threshold_block(None, oof_probs=p, oof_target=y, decision_costs={"fp": 1.0, "fn": 5.0}, n_boot=50)
        assert out["n_positives"] < MIN_POSITIVES_FOR_A_RECOMMENDATION
        assert out["recommended"] is False
        assert "INFORMATIONAL" in format_decision_threshold_line("m", out)

    def test_the_reproducible_seed_gives_a_reproducible_interval(self, rare_positive_oof):
        """A bootstrap interval that moves between runs cannot be quoted in a report."""
        y, p = rare_positive_oof
        kw = dict(oof_probs=p, oof_target=y, decision_costs={"fp": 1.0, "fn": 12.0}, n_boot=60, rng_seed=11)
        assert decision_threshold_block(None, **kw)["threshold_ci"] == decision_threshold_block(None, **kw)["threshold_ci"]


class TestTheLogLine:
    """It has to be readable without opening metadata, and it must not overstate itself."""

    def test_it_names_both_rules_and_says_which_one_is_in_force(self, rare_positive_oof):
        """One line has to carry both thresholds and the fact that 0.5 is still what was scored."""
        y, p = rare_positive_oof
        line = format_decision_threshold_line("bin/y/cb", decision_threshold_block(
            None, oof_probs=p, oof_target=y, decision_costs={"fp": 1.0, "fn": 12.0}, n_boot=50))
        assert "@0.5" in line
        assert "still use 0.5" in line
        assert "CI" in line

    def test_a_skip_says_why(self):
        """A silent skip reads as "tuning found nothing", which is a different claim."""
        assert "skipped" in format_decision_threshold_line("k", decision_threshold_block(None, oof_probs=None, oof_target=None))
