"""Regression test: show_calibration_plot(backend="plotly") must not crash when nothing is requested."""

import os

import numpy as np

from mlframe.metrics.calibration._calibration_plot import show_calibration_plot


def test_plotly_backend_default_args_write_no_file_to_cwd(monkeypatch, tmp_path) -> None:
    """X4 (audits/ci_review_2026-09-08/_TRACKER.md): this exact call -- every argument but backend at
    its default -- used to silently write ``calibration.html`` into the CURRENT WORKING DIRECTORY
    (falling back to the literal base name ``"calibration"`` when ``plot_file`` is empty), which is
    how the stray file reached the repo root and master in commit bd8e212f4. A bare
    ``show_plots=True`` default must not alone count as "save to disk" outside an interactive
    session, matching the matplotlib branch's own no-consumer short-circuit."""
    monkeypatch.chdir(tmp_path)
    freqs_predicted = np.array([0.2, 0.4, 0.6, 0.8])
    freqs_true = np.array([0.22, 0.41, 0.58, 0.79])
    hits = np.array([5, 15, 25, 5])
    result = show_calibration_plot(freqs_predicted, freqs_true, hits, backend="plotly")
    assert result is None
    assert not os.path.exists(tmp_path / "calibration.html"), "show_calibration_plot must not write a file into cwd with every argument at its default"
    assert not list(tmp_path.iterdir()), f"no file at all should have been written; found {list(tmp_path.iterdir())}"


def test_plotly_backend_show_plots_false_no_plot_file_returns_none() -> None:
    """No display/save target requested with backend="plotly" must return None, not crash."""
    freqs_predicted = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    freqs_true = np.array([0.12, 0.28, 0.52, 0.69, 0.88])
    hits = np.array([10, 20, 30, 20, 10])

    # pre-fix: fell through every branch without ever building `fig`, raising
    # UnboundLocalError at the function's final `return fig`.
    result = show_calibration_plot(freqs_predicted, freqs_true, hits, show_plots=False, backend="plotly")
    assert result is None


def test_plotly_backend_default_show_plots_still_returns_none_without_crashing() -> None:
    """Default show_plots with backend="plotly" and no plot_file must not crash."""
    freqs_predicted = np.array([0.2, 0.4, 0.6, 0.8])
    freqs_true = np.array([0.22, 0.41, 0.58, 0.79])
    hits = np.array([5, 15, 25, 5])
    result = show_calibration_plot(freqs_predicted, freqs_true, hits, backend="plotly")
    assert result is None
