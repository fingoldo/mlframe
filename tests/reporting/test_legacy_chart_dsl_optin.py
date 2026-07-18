"""Verify the opt-in DSL render path on the 3 legacy chart functions:

- ``mlframe.metrics.core.show_calibration_plot``
- ``mlframe.training.targets.regression_residual_audit.plot_residual_diagnostics``
- ``mlframe.training.targets.target_temporal_audit.plot_target_over_time``

Each function gained ``plot_outputs`` + ``base_path`` kwargs in 2026-05-08:
when set, the function delegates to the shared spec pipeline
(``build_*_spec`` + ``render_and_save``); when unset, the legacy
matplotlib code path runs unchanged.

Tests assert:
1. Default (no opt-in kwargs) -- legacy path still works (smoke).
2. Opt-in (``plot_outputs="matplotlib[png]"``) -- file emitted at expected
   path, both backends optionally exercised.
3. Plotly backend works (``plot_outputs="plotly[html]"``).
4. Multi-backend DSL (``"matplotlib[png] + plotly[html]"``) emits both.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

# ----------------------------------------------------------------------------
# show_calibration_plot
# ----------------------------------------------------------------------------


@pytest.fixture
def calib_inputs():
    """Calib inputs."""
    rng = np.random.default_rng(0)
    n_bins = 10
    freqs_pred = np.linspace(0.05, 0.95, n_bins)
    freqs_true = freqs_pred + rng.normal(0, 0.05, n_bins)
    hits = rng.integers(50, 1000, n_bins).astype(np.float64)
    return freqs_pred, freqs_true, hits


class TestShowCalibrationPlot:
    """Groups tests for: TestShowCalibrationPlot."""
    def test_legacy_path_unchanged(self, calib_inputs, tmp_path):
        """Legacy path unchanged."""
        from mlframe.metrics.core import show_calibration_plot

        fp, ft, h = calib_inputs
        # No opt-in kwargs -> legacy matplotlib path. Asserts no crash.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = show_calibration_plot(
                fp,
                ft,
                h,
                show_plots=False,
                plot_file=str(tmp_path / "legacy.png"),
            )
        assert fig is not None
        assert os.path.exists(tmp_path / "legacy.png")

    def test_dsl_optin_matplotlib(self, calib_inputs, tmp_path):
        """Dsl option matplotlib."""
        from mlframe.metrics.core import show_calibration_plot

        fp, ft, h = calib_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = show_calibration_plot(
                fp,
                ft,
                h,
                show_plots=False,
                plot_outputs="matplotlib[png]",
                base_path=str(tmp_path / "dsl"),
            )
        # Opt-in path returns None (delegates to render_and_save).
        assert result is None
        assert os.path.exists(tmp_path / "dsl.png")
        assert os.path.getsize(tmp_path / "dsl.png") > 5000

    def test_dsl_optin_plotly(self, calib_inputs, tmp_path):
        """Dsl option plotly."""
        from mlframe.metrics.core import show_calibration_plot

        fp, ft, h = calib_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            show_calibration_plot(
                fp,
                ft,
                h,
                show_plots=False,
                plot_outputs="plotly[html]",
                base_path=str(tmp_path / "dsl"),
            )
        assert os.path.exists(tmp_path / "dsl.html")

    def test_dsl_dual_backend(self, calib_inputs, tmp_path):
        """Dsl dual backend."""
        from mlframe.metrics.core import show_calibration_plot

        fp, ft, h = calib_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            show_calibration_plot(
                fp,
                ft,
                h,
                show_plots=False,
                plot_outputs="matplotlib[png] + plotly[html]",
                base_path=str(tmp_path / "dsl"),
            )
        assert (tmp_path / "dsl.matplotlib.png").exists()
        assert (tmp_path / "dsl.plotly.html").exists()


# ----------------------------------------------------------------------------
# plot_residual_diagnostics
# ----------------------------------------------------------------------------


@pytest.fixture
def reg_inputs():
    """Reg inputs."""
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.standard_normal(n) * 2.0
    y_pred = y_true + rng.standard_normal(n) * 0.5  # well-correlated
    return y_true, y_pred


class TestPlotResidualDiagnostics:
    """Groups tests for: TestPlotResidualDiagnostics."""
    def test_legacy_path_unchanged(self, reg_inputs):
        """Axes-based legacy path still works -- caller supplies axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mlframe.training.targets.regression_residual_audit import plot_residual_diagnostics

        y, yp = reg_inputs
        fig, (ax1, ax2) = plt.subplots(1, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audit = plot_residual_diagnostics(
                y,
                yp,
                ax_hist=ax1,
                ax_resid_vs_pred=ax2,
            )
        assert audit is not None
        plt.close(fig)

    def test_dsl_optin_matplotlib(self, reg_inputs, tmp_path):
        """Dsl option matplotlib."""
        from mlframe.training.targets.regression_residual_audit import plot_residual_diagnostics

        y, yp = reg_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audit = plot_residual_diagnostics(
                y,
                yp,
                plot_outputs="matplotlib[png]",
                base_path=str(tmp_path / "resid"),
                header_str="bizvalue model",
                metrics_str="MAE=0.40 / RMSE=0.50",
            )
        # Opt-in path returns the audit (computed lazily if not supplied).
        assert audit is not None
        assert os.path.exists(tmp_path / "resid.png")
        assert os.path.getsize(tmp_path / "resid.png") > 5000

    def test_dsl_optin_plotly(self, reg_inputs, tmp_path):
        """Dsl option plotly."""
        from mlframe.training.targets.regression_residual_audit import plot_residual_diagnostics

        y, yp = reg_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_residual_diagnostics(
                y,
                yp,
                plot_outputs="plotly[html]",
                base_path=str(tmp_path / "resid"),
            )
        assert os.path.exists(tmp_path / "resid.html")

    def test_degenerate_input_returns_audit_no_crash(self, tmp_path):
        """Degenerate input returns audit no crash."""
        from mlframe.training.targets.regression_residual_audit import plot_residual_diagnostics

        # < 5 finite points -> legacy path returns audit (None); opt-in too.
        y = np.array([1.0, 2.0])
        yp = np.array([1.1, 2.1])
        result = plot_residual_diagnostics(
            y,
            yp,
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "x"),
        )
        # Returns None (no audit precomputed; degenerate input).
        assert result is None
        assert not list(tmp_path.glob("x.*"))


# ----------------------------------------------------------------------------
# plot_target_over_time
# ----------------------------------------------------------------------------


@pytest.fixture
def temporal_audit_result():
    """Build a tiny TemporalAuditResult by calling the real audit."""
    import pandas as pd
    from mlframe.training.targets.target_temporal_audit import audit_target_over_time

    rng = np.random.default_rng(0)
    n = 600
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    target = (rng.random(n) > 0.5).astype(np.int8)
    df = pd.DataFrame({"ts": timestamps, "y": target})
    return audit_target_over_time(
        df,
        timestamp_col="ts",
        target_col="y",
        target_name="y",
        granularity="month",
    )


class TestPlotTargetOverTime:
    """Groups tests for: TestPlotTargetOverTime."""
    def test_legacy_path_unchanged(self, temporal_audit_result, tmp_path):
        """Legacy path unchanged."""
        from mlframe.training.targets.target_temporal_audit import plot_target_over_time

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_target_over_time(
                temporal_audit_result,
                save_path=str(tmp_path / "legacy.png"),
            )
        # Legacy path side-effects: file written.
        assert os.path.exists(tmp_path / "legacy.png")

    def test_dsl_optin_matplotlib(self, temporal_audit_result, tmp_path):
        """Dsl option matplotlib."""
        from mlframe.training.targets.target_temporal_audit import plot_target_over_time

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plot_target_over_time(
                temporal_audit_result,
                plot_outputs="matplotlib[png]",
                base_path=str(tmp_path / "dsl"),
            )
        assert result is None
        assert os.path.exists(tmp_path / "dsl.png")

    def test_dsl_optin_plotly(self, temporal_audit_result, tmp_path):
        """Dsl option plotly."""
        from mlframe.training.targets.target_temporal_audit import plot_target_over_time

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_target_over_time(
                temporal_audit_result,
                plot_outputs="plotly[html]",
                base_path=str(tmp_path / "dsl"),
            )
        assert os.path.exists(tmp_path / "dsl.html")
