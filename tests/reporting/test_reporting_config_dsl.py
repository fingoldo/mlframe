"""Tests for ReportingConfig DSL fields (plot_outputs + multi_target_panels)."""

from __future__ import annotations

import pytest

from mlframe.training.configs import ReportingConfig


class TestPlotOutputsValidation:
    def test_default_is_plotly_html_png(self):
        cfg = ReportingConfig()
        assert cfg.plot_outputs == "plotly[html,png]"

    def test_matplotlib_only_accepted(self):
        cfg = ReportingConfig(plot_outputs="matplotlib[png]")
        assert cfg.plot_outputs == "matplotlib[png]"

    def test_multi_backend_accepted(self):
        cfg = ReportingConfig(plot_outputs="plotly[html] + matplotlib[pdf]")
        assert cfg.plot_outputs == "plotly[html] + matplotlib[pdf]"

    def test_unknown_backend_raises(self):
        with pytest.raises(Exception, match="not supported|backend"):
            ReportingConfig(plot_outputs="bokeh[png]")

    def test_unknown_format_raises(self):
        with pytest.raises(Exception, match="does not support"):
            ReportingConfig(plot_outputs="plotly[gif]")

    def test_matplotlib_html_incompat_raises(self):
        with pytest.raises(Exception, match="does not support"):
            ReportingConfig(plot_outputs="matplotlib[html]")


class TestPanelTemplateValidation:
    def test_multiclass_default(self):
        cfg = ReportingConfig()
        toks = cfg.multiclass_panels.split()
        assert "CONFUSION" in toks
        assert "PR_F1" in toks

    def test_multilabel_default(self):
        cfg = ReportingConfig()
        toks = cfg.multilabel_panels.split()
        assert "PR_F1" in toks
        assert "COOCCURRENCE" in toks

    def test_ltr_default(self):
        cfg = ReportingConfig()
        toks = cfg.ltr_panels.split()
        assert "NDCG_K" in toks
        assert "LIFT" in toks

    def test_unknown_multiclass_token_raises(self):
        with pytest.raises(Exception, match="Unknown multiclass"):
            ReportingConfig(multiclass_panels="CONFUSION FOO")

    def test_unknown_multilabel_token_raises(self):
        with pytest.raises(Exception, match="Unknown multilabel"):
            ReportingConfig(multilabel_panels="PR_F1 BAR")

    def test_unknown_ltr_token_raises(self):
        with pytest.raises(Exception, match="Unknown ltr"):
            ReportingConfig(ltr_panels="NDCG_K BAZ")

    def test_duplicate_token_raises(self):
        with pytest.raises(Exception, match="Duplicate"):
            ReportingConfig(multiclass_panels="CONFUSION CONFUSION")

    def test_empty_template_accepted(self):
        """Empty template = no panels rendered (operator opted out)."""
        cfg = ReportingConfig(multiclass_panels="")
        assert cfg.multiclass_panels == ""

    def test_subset_accepted(self):
        cfg = ReportingConfig(multiclass_panels="CONFUSION ROC")
        assert cfg.multiclass_panels == "CONFUSION ROC"

    def test_extra_panels_accepted(self):
        """``PR_CURVES`` not in default but is in allowed token set."""
        cfg = ReportingConfig(multiclass_panels="CONFUSION PR_CURVES ROC")
        assert "PR_CURVES" in cfg.multiclass_panels
