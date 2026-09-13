"""End-to-end: the DEFAULT ReportingConfig templates dispatch + render the
wave-2/3 panels (NDCG_BY_QSIZE / CONFUSED_PAIRS / COVERAGE / RESID_VS_PRED /
ERR_BY_DECILE) without an explicit per-call override.

These assert the config defaults are genuinely default-ON: a default
``ReportingConfig`` feeds its templates straight into the composer / dispatcher
and the new panels appear, so a future revert of a default template would fail
here -- not silently drop a panel from production reports.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.reporting import render_multi_target_panels
from mlframe.reporting.charts import compose_regression_figure
from mlframe.reporting.spec import FigureSpec
from mlframe.training.configs import ReportingConfig


def _chart_written(directory, stem: str) -> bool:
    """Whether the PNG for ``stem`` was written, under either output layout.

    Charts land in a per-format subfolder by default (``png/name.png``); this test cares that the chart was
    PRODUCED, not which layout the run used.
    """
    return (directory / "png" / f"{stem}.png").exists() or (directory / f"{stem}.png").exists()


def _n_panels(spec: FigureSpec) -> int:
    """Helper: N panels."""
    return sum(1 for row in spec.panels for cell in row if cell is not None)


@pytest.fixture
def mc_inputs():
    """Mc inputs."""
    rng = np.random.default_rng(0)
    n, K = 240, 3
    y = rng.integers(0, K, n)
    proba = rng.dirichlet(alpha=[1] * K, size=n)
    for i, t in enumerate(y):
        proba[i, t] += 0.7
        proba[i] /= proba[i].sum()
    return y, proba, ["cat", "dog", "bird"]


@pytest.fixture
def qr_inputs():
    """Qr inputs."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.standard_normal(n)
    preds = np.column_stack([np.full(n, -1.28), np.zeros(n), np.full(n, 1.28)])
    return y, preds, (0.1, 0.5, 0.9)


class TestDefaultTemplatesDispatch:
    """Groups tests for: TestDefaultTemplatesDispatch."""
    def test_default_multiclass_template_emits_confused_pairs(self, mc_inputs, tmp_path):
        """A default ReportingConfig's ``multiclass_panels`` (which now carries
        CONFUSED_PAIRS) dispatches + renders all its tokens through the real
        auto-dispatch path."""
        y, proba, classes = mc_inputs
        cfg = ReportingConfig()
        assert "CONFUSED_PAIRS" in cfg.multiclass_panels.split()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                probs=proba,
                classes=classes,
                plot_outputs="matplotlib[png]",
                multiclass_panels=cfg.multiclass_panels,
                base_path=str(tmp_path / "mc"),
                target_type="multiclass_classification",
            )
        assert tag == "multiclass"
        assert _chart_written(tmp_path, "mc_multiclass_panels")

    def test_default_quantile_template_emits_coverage(self, qr_inputs, tmp_path):
        """Default quantile template emits coverage."""
        y, preds, alphas = qr_inputs
        cfg = ReportingConfig()
        assert "COVERAGE" in cfg.quantile_panels.split()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y,
                preds=preds,
                quantile_alphas=alphas,
                plot_outputs="matplotlib[png]",
                quantile_panels=cfg.quantile_panels,
                base_path=str(tmp_path / "qr"),
                target_type="quantile_regression",
            )
        assert tag == "quantile"
        assert _chart_written(tmp_path, "qr_quantile_panels")

    def test_default_quantile_template_renders_reliability_decomp_crossing(self, qr_inputs, tmp_path):
        """R-6 end-to-end: the FULL default quantile template -- which now carries QUANTILE_RELIABILITY /
        PINBALL_DECOMP / QUANTILE_CROSSING -- dispatches + renders through the real auto-dispatch path on a default
        suite run. A revert that drops these from ReportingConfig.quantile_panels fails here, not silently."""
        from mlframe.reporting.charts import compose_quantile_figure

        y, preds, alphas = qr_inputs
        cfg = ReportingConfig()
        toks = cfg.quantile_panels.split()
        for tok in ("QUANTILE_RELIABILITY", "PINBALL_DECOMP", "QUANTILE_CROSSING"):
            assert tok in toks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = compose_quantile_figure(y, preds, alphas, panels_template=cfg.quantile_panels)
            tag = render_multi_target_panels(
                targets=y,
                preds=preds,
                quantile_alphas=alphas,
                plot_outputs="matplotlib[png]",
                quantile_panels=cfg.quantile_panels,
                base_path=str(tmp_path / "qrfull"),
                target_type="quantile_regression",
            )
        assert _n_panels(spec) == len(toks)
        assert tag == "quantile"
        assert _chart_written(tmp_path, "qrfull_quantile_panels")

    def test_default_binary_template_renders_pit(self, tmp_path):
        """INV-42 end-to-end: the default binary template now carries PIT and the full template renders through the
        dispatcher. A revert that drops PIT from ReportingConfig.binary_panels fails here."""
        from mlframe.reporting.charts.binary import compose_binary_figure

        rng = np.random.default_rng(0)
        n = 400
        y = rng.integers(0, 2, n)
        score = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.2, n), 0.0, 1.0)
        proba = np.column_stack([1.0 - score, score])
        cfg = ReportingConfig()
        toks = cfg.binary_panels.split()
        assert "PIT" in toks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = compose_binary_figure(y, score, panels_template=cfg.binary_panels)
            tag = render_multi_target_panels(
                targets=y,
                probs=proba,
                plot_outputs="matplotlib[png]",
                binary_panels=cfg.binary_panels,
                base_path=str(tmp_path / "bin"),
                target_type="binary_classification",
            )
        assert _n_panels(spec) == len(toks)
        assert tag == "binary"
        assert _chart_written(tmp_path, "bin_binary_panels")


class TestDefaultRegressionPanels:
    """Default regression reporting is now THREE separate figures (compose_regression_report_figures),
    not one combined grid -- ``ReportingConfig.regression_panels`` stays None to select that default;
    setting it to an explicit token string opts back into the legacy single-figure behaviour."""

    def test_default_regression_panels_config_is_none(self):
        """None is the sentinel for "use the new 3-figure split", not "no panels"."""
        cfg = ReportingConfig()
        assert cfg.regression_panels is None

    def test_default_regression_report_produces_three_figures_without_err_by_decile(self):
        """The default report (panels_template left at its None default) renders predictions / residuals /
        res_dist_and_acf, each with the expected panel count, and ERR_BY_DECILE appears in none of them --
        it was dropped from the default report per explicit user feedback ("не понимаю, выброси его")."""
        from mlframe.reporting.charts import compose_regression_report_figures
        from mlframe.reporting.spec import AnnotationPanelSpec, BarPanelSpec

        rng = np.random.default_rng(0)
        n = 2000
        y = rng.standard_normal(n) * 5.0
        y_pred = y + rng.standard_normal(n) * 0.5
        figures = compose_regression_report_figures(y, y_pred)
        assert set(figures) == {"predictions", "residuals", "res_dist_and_acf"}
        assert _n_panels(figures["predictions"]) == 2
        assert _n_panels(figures["residuals"]) == 2
        assert _n_panels(figures["res_dist_and_acf"]) == 2
        # ERR_BY_DECILE renders as a BarPanelSpec titled "Error by target decile ..."; confirm no panel in
        # any of the three default figures carries that title (the decisive, not just structural, check).
        for fig in figures.values():
            for row in fig.panels:
                for panel in row:
                    if panel is None or isinstance(panel, AnnotationPanelSpec):
                        continue
                    assert not (isinstance(panel, BarPanelSpec) and "target decile" in panel.title.lower())

    def test_legacy_explicit_template_still_renders_one_combined_figure(self):
        """A caller (or a customized ReportingConfig.regression_panels) that explicitly sets a token string
        keeps the OLD single-figure behaviour with exactly the requested panels, unaffected by the new
        default split."""
        rng = np.random.default_rng(0)
        n = 2000
        y = rng.standard_normal(n) * 5.0
        y_pred = y + rng.standard_normal(n) * 0.5
        explicit = "SCATTER RESID_HIST RESID_VS_PRED ERR_BY_DECILE WORM RESID_ACF"
        spec = compose_regression_figure(y, y_pred, panels_template=explicit)
        assert isinstance(spec, FigureSpec)
        assert _n_panels(spec) == len(explicit.split())


class TestDefaultLtrTemplate:
    """Groups tests for: TestDefaultLtrTemplate."""
    def test_default_ltr_template_includes_ndcg_by_qsize(self, tmp_path):
        """Default ltr template includes ndcg by qsize."""
        from mlframe.reporting.charts import compose_ltr_figure

        cfg = ReportingConfig()
        assert "NDCG_BY_QSIZE" in cfg.ltr_panels.split()
        rng = np.random.default_rng(0)
        y, score, gid = [], [], []
        for q in range(40):
            sz = int(rng.integers(4, 9))
            rels = rng.integers(0, 4, sz)
            y.extend(rels.tolist())
            score.extend((rels + rng.normal(0, 0.5, sz)).tolist())
            gid.extend([q] * sz)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = compose_ltr_figure(
                np.asarray(y),
                np.asarray(score, dtype=np.float64),
                np.asarray(gid),
                panels_template=cfg.ltr_panels,
            )
        assert isinstance(spec, FigureSpec)
        assert _n_panels(spec) == len(cfg.ltr_panels.split())
