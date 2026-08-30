"""The post-fit diagnostics block needs ONE budget, not a cap on a single arbitrary member.

A production run skipped the interaction-strength surface because its own 20-second projection exceeded its own
20-second cap, then spent six and a half minutes on the diagnostics that had no cap at all -- longer than the
4m53s model fit the report was describing. A limit that binds on one member of a group and nothing else does
not limit the group; it picks a victim.
"""

from __future__ import annotations

import logging
import time

import pytest

from mlframe.training.reporting._diagnostics_budget import DiagnosticsBudget


class TestTheGate:
    """When it lets a diagnostic through and when it does not."""

    def test_runs_everything_while_there_is_budget(self):
        """A cheap block must be unaffected."""
        budget = DiagnosticsBudget(60.0)
        assert [budget.run(str(i), lambda i=i: i) for i in range(3)] == [0, 1, 2]
        assert budget.skipped == []

    def test_skips_once_exhausted_and_names_what_it_skipped(self):
        """A shortened report must never look like a complete one."""
        budget = DiagnosticsBudget(0.01)
        time.sleep(0.02)
        assert budget.run("pdp_ice", lambda: "drawn") is None
        assert budget.skipped == ["pdp_ice"]

    def test_zero_disables_the_budget(self):
        """The escape hatch for a caller who wants the full report regardless of cost."""
        budget = DiagnosticsBudget(0.0)
        time.sleep(0.01)
        assert budget.exhausted() is False
        assert budget.run("shap", lambda: "drawn") == "drawn"

    @pytest.mark.parametrize("value", [None, 0, -1.0])
    def test_missing_or_negative_budget_disables_it(self, value):
        """An unset knob must not silently truncate the report."""
        assert DiagnosticsBudget(value or 0.0).exhausted() is False

    def test_a_running_diagnostic_is_never_interrupted(self):
        """The gate is checked BETWEEN diagnostics: a half-drawn figure is worse than a missing one."""
        budget = DiagnosticsBudget(0.05)
        result = budget.run("slow", lambda: (time.sleep(0.12), "finished")[1])
        assert result == "finished"
        assert budget.skipped == []
        assert budget.exhausted() is True


class TestTheReport:
    """What the operator is told."""

    def test_nothing_skipped_says_nothing(self, caplog):
        """A complete report must not carry a warning about completeness."""
        budget = DiagnosticsBudget(60.0)
        budget.run("a", lambda: None)
        with caplog.at_level(logging.WARNING, logger="mlframe.training.reporting._diagnostics_budget"):
            budget.report()
        assert caplog.records == []

    def test_skips_are_reported_once_with_names_and_the_knob(self, caplog):
        """One line, naming every dropped diagnostic and how to get them back."""
        budget = DiagnosticsBudget(0.01)
        time.sleep(0.02)
        for name in ("pdp_ice", "shap", "slice_finder"):
            budget.run(name, lambda: None)
        with caplog.at_level(logging.WARNING, logger="mlframe.training.reporting._diagnostics_budget"):
            budget.report()
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "INCOMPLETE" in text
        assert "pdp_ice, shap, slice_finder" in text
        assert "diagnostics_max_seconds" in text


class TestTheConfigKnob:
    """The default has to be larger than a normal block and smaller than the runaway one."""

    def test_default_exists_and_is_a_real_budget(self):
        """0 would mean no budget at all, which is the state this fix exists to end."""
        from mlframe.training._reporting_configs import ReportingConfig

        assert ReportingConfig().diagnostics_max_seconds > 0


class TestHeavyDiagnosticsScope:
    """Redrawing SHAP and PDP for five aggregations of the same two members describes one model five times.

    The production run did exactly that: five ensemble flavours of two members correlated at 0.996, every one
    returning the same test AUC and Brier to two decimals. Scoping the model-explanation diagnostics to the
    primary model is a change of SCOPE; the metric and calibration panels, which exist to compare members,
    still render for all of them.
    """

    def _budget(self, mode: str, is_primary: bool):
        """A budget with no time limit, so only the scope policy can decide anything."""
        from mlframe.training.reporting._diagnostics_budget import DiagnosticsBudget, HeavyDiagnosticsPolicy

        return DiagnosticsBudget(0.0, policy=HeavyDiagnosticsPolicy(mode=mode, is_primary=is_primary))

    @pytest.mark.parametrize("name", ["shap", "pdp_ice", "slice_finder", "interaction_strength"])
    def test_heavy_diagnostics_skipped_for_a_variant(self, name):
        """The whole point: an ensemble variant does not get its own SHAP surface."""
        budget = self._budget("best", is_primary=False)
        assert budget.run(name, lambda: "drawn") is None
        assert budget.out_of_scope == [name]

    @pytest.mark.parametrize("name", ["decision_curve", "decile_table", "model_card", "risk_coverage"])
    def test_comparison_panels_still_render_for_a_variant(self, name):
        """These are how a reader tells the variants apart, so scoping them away would defeat the report."""
        assert self._budget("best", is_primary=False).run(name, lambda: "drawn") == "drawn"

    @pytest.mark.parametrize("name", ["shap", "pdp_ice"])
    def test_primary_model_gets_everything(self, name):
        """The model the report is actually about keeps its full explanation set."""
        assert self._budget("best", is_primary=True).run(name, lambda: "drawn") == "drawn"

    def test_all_mode_restores_the_previous_behaviour(self):
        """A caller who wants a surface per variant must be able to have one."""
        assert self._budget("all", is_primary=False).run("shap", lambda: "drawn") == "drawn"

    def test_out_of_scope_is_reported_separately_from_a_budget_skip(self, caplog):
        """ "Not applicable here" and "we ran out of time" are different facts and must not read alike."""
        budget = self._budget("best", is_primary=False)
        budget.run("shap", lambda: None)
        with caplog.at_level(logging.INFO, logger="mlframe.training.reporting._diagnostics_budget"):
            budget.report()
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "restricted to the primary model" in text
        assert "INCOMPLETE" not in text

    def test_default_mode_is_best(self):
        """The default has to be the one that stops the waste; "all" is the opt-in."""
        from mlframe.training._reporting_configs import ReportingConfig

        assert ReportingConfig().heavy_diagnostics_for == "best"
