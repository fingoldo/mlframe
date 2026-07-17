"""Tests for the single-source reliability overlay spec + policy routing.

Covers INV-38 / INV-40: build_calibration_spec / build_reliability_overlay_spec
are the one source for the reliability diagram; calibration.policy._emit_reliability_plot
is expressed through the spec (LinePanelSpec overlay of raw vs calibrated curves).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import numpy as np

from mlframe.reporting.charts.calibration import build_reliability_overlay_spec
from mlframe.reporting.spec import FigureSpec, LinePanelSpec


def test_overlay_spec_shape_and_series():
    rng = np.random.default_rng(0)
    n = 2000
    raw = rng.random(n)
    y = (rng.random(n) < raw).astype(np.float64)
    calibrated = {"Iso": np.clip(raw + 0.05, 0, 1), "Beta": np.clip(raw - 0.03, 0, 1)}
    labels = {"Iso": "Iso ECE=0.0100", "Beta": "Beta ECE=0.0200"}
    spec = build_reliability_overlay_spec(
        raw,
        y,
        calibrated_probs=calibrated,
        series_labels=labels,
        n_bins=15,
    )
    assert isinstance(spec, FigureSpec)
    panel = spec.panels[0][0]
    assert isinstance(panel, LinePanelSpec)
    # perfect + raw + 2 candidates = 4 series.
    assert len(panel.y) == 4
    assert panel.series_labels == ("perfect", "raw OOF", "Iso ECE=0.0100", "Beta ECE=0.0200")
    # First series is the perfect diagonal (== x centers).
    np.testing.assert_allclose(panel.y[0], panel.x, rtol=1e-12)
    # All curves share the 15-bin centre grid.
    assert len(panel.x) == 15


def test_overlay_curve_observed_freq_matches_reference():
    """The raw-OOF curve y equals the observed positive rate per uniform bin."""
    n_bins = 10
    # Perfectly calibrated synthetic: observed freq per bin ~ bin centre.
    rng = np.random.default_rng(2)
    n = 50000
    raw = rng.random(n)
    y = (rng.random(n) < raw).astype(np.float64)
    spec = build_reliability_overlay_spec(raw, y, n_bins=n_bins)
    raw_curve = spec.panels[0][0].y[1]  # series index 1 = raw OOF
    centers = spec.panels[0][0].x
    # On a well-calibrated source the observed rate tracks the bin centre.
    finite = np.isfinite(raw_curve)
    assert np.max(np.abs(raw_curve[finite] - centers[finite])) < 0.05


def test_emit_reliability_plot_routes_through_spec(tmp_path, monkeypatch):
    """policy._emit_reliability_plot must build a FigureSpec and hand it to the
    shared renderer (INV-38), not draw its own matplotlib axes."""
    import mlframe.calibration.policy as policy
    import mlframe.reporting.renderers as renderers

    captured = {}

    def _fake_render_and_save(spec, output, base_path, **kw):
        captured["spec"] = spec
        captured["base_path"] = base_path
        # Write a stub file so the function's path-return contract still holds.
        with open(base_path + ".png", "wb") as fh:
            fh.write(b"\x89PNG" + b"0" * 2048)
        return None

    monkeypatch.setattr(renderers, "render_and_save", _fake_render_and_save)

    rng = np.random.default_rng(1)
    n = 1000
    raw = rng.random(n)
    y = (rng.random(n) < raw).astype(np.float64)
    candidates = {
        "Iso": {"calibrated_probs": np.clip(raw + 0.02, 0, 1), "ece_mean": 0.012},
    }
    out_path = str(tmp_path / "calib.png")
    result = policy._emit_reliability_plot(candidates, raw, y, out_path, n_bins=15)

    assert result is not None
    assert isinstance(captured["spec"], FigureSpec)
    panel = captured["spec"].panels[0][0]
    assert isinstance(panel, LinePanelSpec)
    # perfect + raw + Iso = 3 series.
    assert len(panel.y) == 3
    assert os.path.exists(result)
