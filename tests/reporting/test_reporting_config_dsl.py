"""Tests for ReportingConfig DSL fields (plot_outputs + multi_target_panels)."""

from __future__ import annotations

import pytest

from mlframe.training.configs import ReportingConfig


class TestPlotOutputsValidation:
    def test_default_is_plotly_html_plus_matplotlib_png(self):
        # 2026-05-10: default flipped from "plotly[html,png]" because the
        # PNG-via-plotly path goes through kaleido (Chromium page.reload()
        # ~12-15 s per figure). New default keeps interactive HTML
        # (plotly) + matplotlib PNG (10-20x faster, no Chromium).
        cfg = ReportingConfig()
        assert cfg.plot_outputs == "plotly[html] + matplotlib[png]"

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

    def test_quantile_default(self):
        cfg = ReportingConfig()
        toks = cfg.quantile_panels.split()
        assert "RELIABILITY" in toks
        assert "PIT_HIST" in toks

    def test_regression_default(self):
        cfg = ReportingConfig()
        toks = cfg.regression_panels.split()
        assert "SCATTER" in toks
        assert "RESID_HIST" in toks

    def test_unknown_quantile_token_raises(self):
        with pytest.raises(Exception, match="Unknown quantile"):
            ReportingConfig(quantile_panels="RELIABILITY ZZZ")

    def test_unknown_regression_token_raises(self):
        with pytest.raises(Exception, match="Unknown regression"):
            ReportingConfig(regression_panels="SCATTER ZZZ")


class TestNewPanelsDefaultOn:
    """Each wave-2/3 panel must be (a) in its target type's default template
    and (b) accepted by the validator. The validator now sources its allowed
    sets directly from the chart modules' frozensets, so a builder dropping a
    token would trip both halves here."""

    def test_ndcg_by_qsize_default_on_and_valid(self):
        cfg = ReportingConfig()
        assert "NDCG_BY_QSIZE" in cfg.ltr_panels.split()
        assert ReportingConfig(ltr_panels="NDCG_BY_QSIZE").ltr_panels == "NDCG_BY_QSIZE"

    def test_confused_pairs_default_on_and_valid(self):
        cfg = ReportingConfig()
        assert "CONFUSED_PAIRS" in cfg.multiclass_panels.split()
        assert ReportingConfig(multiclass_panels="CONFUSED_PAIRS").multiclass_panels == "CONFUSED_PAIRS"

    def test_coverage_default_on_and_valid(self):
        cfg = ReportingConfig()
        assert "COVERAGE" in cfg.quantile_panels.split()
        assert ReportingConfig(quantile_panels="COVERAGE").quantile_panels == "COVERAGE"

    def test_resid_vs_pred_default_on_and_valid(self):
        cfg = ReportingConfig()
        assert "RESID_VS_PRED" in cfg.regression_panels.split()
        assert ReportingConfig(regression_panels="RESID_VS_PRED").regression_panels == "RESID_VS_PRED"

    def test_err_by_decile_default_on_and_valid(self):
        cfg = ReportingConfig()
        assert "ERR_BY_DECILE" in cfg.regression_panels.split()
        assert ReportingConfig(regression_panels="ERR_BY_DECILE").regression_panels == "ERR_BY_DECILE"

    def test_quantile_reliability_decomp_crossing_default_on(self):
        """R-6: the three quantile-reliability tokens must be in the SUITE default ReportingConfig.quantile_panels
        (not just the composer's DEFAULT_QUANTILE_PANELS), since the suite passes the config value to the dispatcher.
        Pre-fix the default omitted all three so they never rendered on a default suite run."""
        toks = ReportingConfig().quantile_panels.split()
        for tok in ("QUANTILE_RELIABILITY", "PINBALL_DECOMP", "QUANTILE_CROSSING"):
            assert tok in toks, f"{tok} must be default-ON in ReportingConfig.quantile_panels"

    def test_config_quantile_default_is_composer_default_plus_fan_chart(self):
        """The suite default is the composer's conservative DEFAULT_QUANTILE_PANELS plus FAN_CHART, which the
        integrator enables default-on at the suite level (it is valid but kept out of the library composer default)."""
        from mlframe.reporting.charts.quantile import DEFAULT_QUANTILE_PANELS

        cfg_toks = set(ReportingConfig().quantile_panels.split())
        composer_toks = set(DEFAULT_QUANTILE_PANELS.split())
        assert composer_toks <= cfg_toks, "every composer-default token must stay default-on in the suite config"
        assert cfg_toks - composer_toks == {"FAN_CHART"}, "the only suite-added quantile token is FAN_CHART"

    def test_fan_chart_default_on_in_quantile(self):
        assert "FAN_CHART" in ReportingConfig().quantile_panels.split()

    def test_worm_resid_acf_default_on_in_regression(self):
        toks = ReportingConfig().regression_panels.split()
        assert "WORM" in toks and "RESID_ACF" in toks

    def test_threshold_sweep_default_on_in_multilabel(self):
        assert "THRESHOLD_SWEEP" in ReportingConfig().multilabel_panels.split()

    def test_pit_default_on_in_binary_and_valid(self):
        """INV-42: PIT must be in the binary default template (the token + _pit_panel already exist)."""
        assert "PIT" in ReportingConfig().binary_panels.split()
        assert ReportingConfig(binary_panels="PIT").binary_panels == "PIT"

    def test_validator_allowed_sets_match_chart_frozensets(self):
        """The validator must accept exactly the chart modules' frozensets --
        every token in each ALLOWED_*_PANEL_TOKENS round-trips through its
        field, proving the validator is sourced from the single source of
        truth (not a drifted literal copy)."""
        from mlframe.reporting.charts import (
            ALLOWED_LTR_PANEL_TOKENS,
            ALLOWED_MULTICLASS_PANEL_TOKENS,
            ALLOWED_MULTILABEL_PANEL_TOKENS,
            ALLOWED_QUANTILE_PANEL_TOKENS,
            ALLOWED_REGRESSION_PANEL_TOKENS,
        )
        for field, allowed in (
            ("multiclass_panels", ALLOWED_MULTICLASS_PANEL_TOKENS),
            ("multilabel_panels", ALLOWED_MULTILABEL_PANEL_TOKENS),
            ("ltr_panels", ALLOWED_LTR_PANEL_TOKENS),
            ("quantile_panels", ALLOWED_QUANTILE_PANEL_TOKENS),
            ("regression_panels", ALLOWED_REGRESSION_PANEL_TOKENS),
        ):
            template = " ".join(sorted(allowed))
            cfg = ReportingConfig(**{field: template})
            assert getattr(cfg, field) == template


class TestCalibrationAndReliabilityKnobs:
    def test_calibration_binning_default_auto(self):
        assert ReportingConfig().calibration_binning == "auto"

    def test_calibration_binning_accepts_uniform_quantile(self):
        assert ReportingConfig(calibration_binning="uniform").calibration_binning == "uniform"
        assert ReportingConfig(calibration_binning="quantile").calibration_binning == "quantile"

    def test_calibration_binning_rejects_unknown(self):
        with pytest.raises(Exception):
            ReportingConfig(calibration_binning="bogus")

    def test_reliability_show_ci_default_on(self):
        assert ReportingConfig().reliability_show_ci is True

    def test_reliability_show_ci_can_disable(self):
        assert ReportingConfig(reliability_show_ci=False).reliability_show_ci is False


class TestRegressionScatterSampleKnob:
    """INV-15: the suite-default regression scatter sample must be 5000, not the historical 500."""

    def test_default_regression_scatter_sample_is_5000(self):
        assert ReportingConfig().regression_scatter_sample_size == 5000

    def test_regression_scatter_sample_is_overridable(self):
        assert ReportingConfig(regression_scatter_sample_size=20000).regression_scatter_sample_size == 20000

    def test_default_plot_sample_size_constant_is_5000(self):
        """The runtime default the suite path resolves to when no config is threaded must also be 5000
        (pre-fix this constant was 500 and overrode the spec builder's 5000)."""
        from mlframe.training.reporting._reporting import DEFAULT_PLOT_SAMPLE_SIZE

        assert DEFAULT_PLOT_SAMPLE_SIZE == 5000


class TestKeepFigureHandlesField:
    """INV-56: keep_figure_handles must be a real ReportingConfig field (the render path reads it via getattr)."""

    def test_keep_figure_handles_field_exists_default_false(self):
        assert "keep_figure_handles" in ReportingConfig.model_fields
        assert ReportingConfig().keep_figure_handles is False

    def test_keep_figure_handles_can_enable(self):
        assert ReportingConfig(keep_figure_handles=True).keep_figure_handles is True
