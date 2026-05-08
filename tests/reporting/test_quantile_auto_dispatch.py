"""Tests for the QR branch of ``render_multi_target_panels``.

QR dispatcher fires when (preds is 2-D + targets 1-D + quantile_alphas
+ quantile_panels are all set). Verifies:
- Dispatch routes to QR composer + writes ``*_quantile_panels.{ext}``
- Missing ``quantile_alphas`` -> no-op (back-compat: regression caller
  with 1-D preds doesn't accidentally hit QR path)
- Multi-backend DSL emits both files
- Composer-side exceptions logged + swallowed
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from mlframe.reporting import render_multi_target_panels


@pytest.fixture
def qr_inputs():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.standard_normal(n)
    preds = np.column_stack([
        np.full(n, -1.28),
        np.zeros(n),
        np.full(n, 1.28),
    ])
    return y, preds, (0.1, 0.5, 0.9)


class TestQRDispatch:
    def test_quantile_dispatch(self, qr_inputs, tmp_path):
        y, preds, alphas = qr_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tag = render_multi_target_panels(
                targets=y, preds=preds, quantile_alphas=alphas,
                plot_outputs="matplotlib[png]",
                quantile_panels="RELIABILITY WIDTH_DIST",
                base_path=str(tmp_path / "qr"),
            )
        assert tag == "quantile"
        assert os.path.exists(tmp_path / "qr_quantile_panels.png")

    def test_no_alphas_no_dispatch(self, qr_inputs, tmp_path):
        y, preds, _ = qr_inputs
        # quantile_panels set but quantile_alphas missing -> falls through
        # to multilabel/multiclass branches (which won't match 1-D y +
        # 2-D preds without classes), then returns None.
        tag = render_multi_target_panels(
            targets=y, preds=preds,
            plot_outputs="matplotlib[png]",
            quantile_panels="RELIABILITY",
            base_path=str(tmp_path / "qr"),
        )
        assert tag is None

    def test_no_quantile_panels_template_no_dispatch(self, qr_inputs, tmp_path):
        y, preds, alphas = qr_inputs
        tag = render_multi_target_panels(
            targets=y, preds=preds, quantile_alphas=alphas,
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "qr"),
        )
        assert tag is None

    def test_dual_backend(self, qr_inputs, tmp_path):
        y, preds, alphas = qr_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            render_multi_target_panels(
                targets=y, preds=preds, quantile_alphas=alphas,
                plot_outputs="matplotlib[png] + plotly[html]",
                quantile_panels="RELIABILITY",
                base_path=str(tmp_path / "qr"),
            )
        assert (tmp_path / "qr_quantile_panels.matplotlib.png").exists()
        assert (tmp_path / "qr_quantile_panels.plotly.html").exists()

    def test_composer_exception_swallowed(self, qr_inputs, tmp_path, monkeypatch, caplog):
        import logging
        from mlframe.reporting import auto_dispatch
        import mlframe.reporting.charts.quantile as qmod

        def _raise(*args, **kwargs):
            raise RuntimeError("synthetic")

        monkeypatch.setattr(qmod, "compose_quantile_figure", _raise)
        y, preds, alphas = qr_inputs
        with caplog.at_level(logging.ERROR, logger=auto_dispatch.logger.name):
            tag = render_multi_target_panels(
                targets=y, preds=preds, quantile_alphas=alphas,
                plot_outputs="matplotlib[png]",
                quantile_panels="RELIABILITY",
                base_path=str(tmp_path / "qr"),
            )
        # Composer raised -> dispatcher logs + falls through; returns None.
        assert tag is None
        assert any("Quantile panel rendering failed" in r.getMessage()
                   for r in caplog.records)
