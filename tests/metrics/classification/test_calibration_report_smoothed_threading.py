"""Threading proof: the binning-free smoothed isotonic reliability overlay is default-ON in suite reliability diagrams.

The overlay is composed inside ``build_calibration_spec`` from the raw per-row ``(raw_probs, raw_labels)`` pairs. These
tests prove the SUITE call chain (``report_probabilistic_model_perf`` -> ``fast_calibration_report`` ->
``show_calibration_plot`` -> ``build_calibration_spec``) threads those raw arrays down end-to-end via the DSL render
path, so the overlay actually reaches the figure spec -- not just when ``build_calibration_spec`` is called directly.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.reporting.charts.calibration as cal_mod
from mlframe.metrics.classification._classification_report import fast_calibration_report
from mlframe.reporting.spec import ScatterPanelSpec


def _overconfident(n: int = 5000, seed: int = 0):
    """Helper: Overconfident."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=n)
    p_true = 1.0 / (1.0 + np.exp(-z))
    score = 1.0 / (1.0 + np.exp(-2.0 * z))
    y = (rng.random(n) < p_true).astype(np.int64)
    return y, score


@pytest.fixture
def _spec_spy(monkeypatch):
    """Wrap build_calibration_spec to capture the spec it returns + the kwargs it received, while still rendering."""
    captured = {}
    real = cal_mod.build_calibration_spec

    def _spy(*args, **kwargs):
        """Helper: Spy."""
        spec = real(*args, **kwargs)
        captured["kwargs"] = kwargs
        captured["spec"] = spec
        return spec

    monkeypatch.setattr(cal_mod, "build_calibration_spec", _spy)
    return captured


def _overlay_from_spec(spec):
    """Helper: Overlay from spec."""
    scatter = spec.panels[0][0]
    assert isinstance(scatter, ScatterPanelSpec)
    return scatter.overlay_line


class TestSuiteThreadingDefaultOn:
    """Groups tests for: TestSuiteThreadingDefaultOn."""
    def test_fast_calibration_report_threads_overlay_via_dsl(self, _spec_spy, tmp_path):
        """A suite-style fast_calibration_report call with the DSL render path produces a spec whose reliability panel
        CARRIES the smoothed overlay -- raw arrays threaded end-to-end with no explicit reliability_smoothed flag."""
        y, score = _overconfident()
        fast_calibration_report(
            y_true=y,
            y_pred=score,
            nbins=15,
            show_plots=False,
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "cal"),
        )
        kw = _spec_spy["kwargs"]
        # Raw per-row arrays threaded down (the SAME object, no copy) so the smoother subsamples internally.
        assert kw.get("raw_probs") is score
        assert kw.get("raw_labels") is y
        assert kw.get("reliability_smoothed") is True
        overlay = _overlay_from_spec(_spec_spy["spec"])
        assert overlay is not None, "smoothed overlay must reach the figure spec on the suite DSL path"
        assert "isotonic" in overlay[2]

    def test_overlay_default_on_without_explicit_flag(self, _spec_spy, tmp_path):
        """Default-ON: no reliability_smoothed passed anywhere -> overlay still present on the suite path."""
        y, score = _overconfident(seed=1)
        fast_calibration_report(
            y_true=y,
            y_pred=score,
            nbins=10,
            show_plots=False,
            plot_outputs="plotly[html]",
            base_path=str(tmp_path / "cal2"),
        )
        assert _overlay_from_spec(_spec_spy["spec"]) is not None

    def test_toggle_off_suppresses_overlay_through_chain(self, _spec_spy, tmp_path):
        """reliability_smoothed=False at the report boundary suppresses the overlay end-to-end."""
        y, score = _overconfident(seed=2)
        fast_calibration_report(
            y_true=y,
            y_pred=score,
            nbins=15,
            show_plots=False,
            reliability_smoothed=False,
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "cal3"),
        )
        assert _spec_spy["kwargs"].get("reliability_smoothed") is False
        assert _overlay_from_spec(_spec_spy["spec"]) is None

    def test_prebinned_only_call_still_works_without_overlay(self):
        """The pre-binned-only composer call (no raw arrays) must still build a valid spec, overlay simply absent."""
        spec = cal_mod.build_calibration_spec(
            np.linspace(0.05, 0.95, 10),
            np.linspace(0.0, 1.0, 10),
            np.full(10, 500),
        )
        assert _overlay_from_spec(spec) is None


class TestReportProbabilisticThreading:
    """Groups tests for: TestReportProbabilisticThreading."""
    def test_report_perf_threads_overlay_per_class(self, _spec_spy, tmp_path):
        """report_probabilistic_model_perf (binary) routes its per-class one-vs-rest raw arrays through the chain so
        the suite reliability diagram carries the smoothed overlay by default."""
        from mlframe.training.reporting._reporting_probabilistic import report_probabilistic_model_perf

        y, score = _overconfident(seed=3)
        probs = np.column_stack([1.0 - score, score])
        report_probabilistic_model_perf(
            targets=y,
            columns=["f0"],
            model_name="m",
            model=None,
            probs=probs,
            classes=[0, 1],
            show_perf_chart=False,
            print_report=False,
            plot_file=str(tmp_path / "perf"),
            plot_outputs="matplotlib[png]",
        )
        # Binary path reports only class_id=1; raw_probs is that positive-class score column (a view), labels are y==1.
        kw = _spec_spy["kwargs"]
        assert kw.get("reliability_smoothed") is True
        assert kw.get("raw_probs") is not None and kw.get("raw_labels") is not None
        overlay = _overlay_from_spec(_spec_spy["spec"])
        assert overlay is not None and "isotonic" in overlay[2]
