"""End-to-end: the DEFAULT ReportingConfig templates dispatch + render the
wave-2/3 panels (NDCG_BY_QSIZE / CONFUSED_PAIRS / COVERAGE / RESID_VS_PRED /
ERR_BY_DECILE) without an explicit per-call override.

These assert the config defaults are genuinely default-ON: a default
``ReportingConfig`` feeds its templates straight into the composer / dispatcher
and the new panels appear, so a future revert of a default template would fail
here -- not silently drop a panel from production reports.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mlframe.reporting import render_multi_target_panels
from mlframe.reporting.charts import compose_regression_figure
from mlframe.reporting.spec import FigureSpec
from mlframe.training.configs import ReportingConfig


def _n_panels(spec: FigureSpec) -> int:
    return sum(1 for row in spec.panels for cell in row if cell is not None)


@pytest.fixture
def mc_inputs():
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
    rng = np.random.default_rng(0)
    n = 300
    y = rng.standard_normal(n)
    preds = np.column_stack([np.full(n, -1.28), np.zeros(n), np.full(n, 1.28)])
    return y, preds, (0.1, 0.5, 0.9)


class TestDefaultTemplatesDispatch:
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
                targets=y, probs=proba, classes=classes,
                plot_outputs="matplotlib[png]",
                multiclass_panels=cfg.multiclass_panels,
                base_path=str(tmp_path / "mc"),
                target_type="multiclass_classification",
            )
        assert tag == "multiclass"
        assert os.path.exists(tmp_path / "mc_multiclass_panels.png")

    def test_default_quantile_template_emits_coverage(self, qr_inputs, tmp_path):
        y, preds, alphas = qr_inputs
        cfg = ReportingConfig()
        assert "COVERAGE" in cfg.quantile_panels.split()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y, preds=preds, quantile_alphas=alphas,
                plot_outputs="matplotlib[png]",
                quantile_panels=cfg.quantile_panels,
                base_path=str(tmp_path / "qr"),
                target_type="quantile_regression",
            )
        assert tag == "quantile"
        assert os.path.exists(tmp_path / "qr_quantile_panels.png")

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
                targets=y, preds=preds, quantile_alphas=alphas,
                plot_outputs="matplotlib[png]",
                quantile_panels=cfg.quantile_panels,
                base_path=str(tmp_path / "qrfull"),
                target_type="quantile_regression",
            )
        assert _n_panels(spec) == len(toks)
        assert tag == "quantile"
        assert os.path.exists(tmp_path / "qrfull_quantile_panels.png")

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
                targets=y, probs=proba,
                plot_outputs="matplotlib[png]",
                binary_panels=cfg.binary_panels,
                base_path=str(tmp_path / "bin"),
                target_type="binary_classification",
            )
        assert _n_panels(spec) == len(toks)
        assert tag == "binary"
        assert os.path.exists(tmp_path / "bin_binary_panels.png")


class TestDefaultRegressionPanels:
    def test_default_regression_template_includes_new_panels(self):
        cfg = ReportingConfig()
        toks = cfg.regression_panels.split()
        assert "RESID_VS_PRED" in toks
        assert "ERR_BY_DECILE" in toks

    def test_default_regression_composer_renders_one_panel_per_token(self):
        """The regression composer renders one panel per token in the config default, including the
        RESID_VS_PRED + decile panels and the integrator-enabled WORM + RESID_ACF (the report path uses this default)."""
        rng = np.random.default_rng(0)
        n = 2000
        y = rng.standard_normal(n) * 5.0
        y_pred = y + rng.standard_normal(n) * 0.5
        cfg = ReportingConfig()
        spec = compose_regression_figure(y, y_pred, panels_template=cfg.regression_panels)
        assert isinstance(spec, FigureSpec)
        assert _n_panels(spec) == len(cfg.regression_panels.split())


class TestDefaultLtrTemplate:
    def test_default_ltr_template_includes_ndcg_by_qsize(self, tmp_path):
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
                np.asarray(y), np.asarray(score, dtype=np.float64), np.asarray(gid),
                panels_template=cfg.ltr_panels,
            )
        assert isinstance(spec, FigureSpec)
        assert _n_panels(spec) == len(cfg.ltr_panels.split())
