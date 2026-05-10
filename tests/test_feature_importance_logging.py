"""Tests for ``mlframe.feature_importance.plot_feature_importance``:

* **inline display fix (2026-05-11)** -- the renderer-induced Agg
  backend mutation in ``reporting/renderers/matplotlib.py`` made
  ``plt.show()`` a no-op inside Jupyter; the FI plotter now uses
  ``IPython.display.display(fig)`` when ``__IPYTHON__`` is defined so
  inline display works regardless of the global matplotlib backend.
* **top-N text log (2026-05-11)** -- new ``log_top_n`` / ``log_fi``
  knobs default-ON emit a compact one-paragraph string of the top-N
  feature importances via ``logger.info`` whether or not a plot is
  rendered.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # keep CI headless; tests must NOT call plt.show

import numpy as np
import pytest

from mlframe.feature_importance import (
    plot_feature_importance,
    _format_top_fi_for_log,
)


# ----------------------------------------------------------------------
# Top-N text log
# ----------------------------------------------------------------------


class TestTopNTextLog:
    def test_format_top_fi_includes_names_and_values(self) -> None:
        """The formatter emits `feature_name=value` pairs in descending
        order."""
        import pandas as pd
        sorted_df = pd.DataFrame(
            {"fi": [0.5, 0.3, 0.1, 0.05]},
            index=["alpha", "beta", "gamma", "delta"],
        )
        text = _format_top_fi_for_log(
            sorted_df=sorted_df, top_n=3, kind="MODEL",
        )
        # Names + values for top-3.
        assert "alpha=" in text and "0.5" in text
        assert "beta=" in text and "0.3" in text
        assert "gamma=" in text and "0.1" in text
        # 4th feature (delta) NOT included (top_n=3).
        assert "delta" not in text
        # Header includes model kind.
        assert "MODEL" in text
        # Summary line includes total feature count.
        assert "N_total=4" in text

    def test_logger_emits_top_n_when_log_fi_true(self, caplog) -> None:
        """``log_fi=True`` (default) emits the top-N FI line via
        ``logger.info``."""
        fi = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        columns = ["f0", "f1", "f2", "f3", "f4"]
        with caplog.at_level(logging.INFO,
                              logger="mlframe.feature_importance"):
            plot_feature_importance(
                feature_importances=fi, columns=columns,
                kind="UNIT_TEST_KIND",
                show_plots=False, plot_file="",
                log_top_n=3, log_fi=True,
            )
        # Find the FI top-N log line.
        fi_lines = [
            r.message for r in caplog.records
            if "[FI top-" in r.message
        ]
        assert fi_lines, "expected a '[FI top-N]' log line"
        line = fi_lines[0]
        assert "UNIT_TEST_KIND" in line
        # Top 3 features by FI: f0(.4), f1(.3), f2(.2).
        assert "f0=" in line and "f1=" in line and "f2=" in line

    def test_logger_silent_when_log_fi_false(self, caplog) -> None:
        """``log_fi=False`` suppresses the log line even with
        ``log_top_n > 0``."""
        fi = np.array([0.4, 0.3, 0.2])
        columns = ["a", "b", "c"]
        with caplog.at_level(logging.INFO,
                              logger="mlframe.feature_importance"):
            plot_feature_importance(
                feature_importances=fi, columns=columns,
                kind="SILENT", show_plots=False, plot_file="",
                log_top_n=10, log_fi=False,
            )
        assert not any(
            "[FI top-" in r.message for r in caplog.records
        )

    def test_logger_silent_when_log_top_n_zero(self, caplog) -> None:
        """``log_top_n=0`` suppresses the log line."""
        fi = np.array([0.4, 0.3, 0.2])
        columns = ["a", "b", "c"]
        with caplog.at_level(logging.INFO,
                              logger="mlframe.feature_importance"):
            plot_feature_importance(
                feature_importances=fi, columns=columns,
                kind="ZERO_TOP", show_plots=False, plot_file="",
                log_top_n=0, log_fi=True,
            )
        assert not any(
            "[FI top-" in r.message for r in caplog.records
        )

    def test_default_log_fi_is_on(self, caplog) -> None:
        """Per user's request: text logging default-ON."""
        fi = np.array([0.5, 0.3, 0.2])
        columns = ["x", "y", "z"]
        with caplog.at_level(logging.INFO,
                              logger="mlframe.feature_importance"):
            # No log_fi / log_top_n -- use defaults.
            plot_feature_importance(
                feature_importances=fi, columns=columns,
                kind="DEFAULT_ON",
                show_plots=False, plot_file="",
            )
        assert any(
            "[FI top-" in r.message and "DEFAULT_ON" in r.message
            for r in caplog.records
        ), "default log_fi should emit the FI log line"


# ----------------------------------------------------------------------
# Inline display branch (Jupyter detection)
# ----------------------------------------------------------------------


class TestInlineDisplayPath:
    def test_no_jupyter_falls_through_to_plt_show(self, monkeypatch) -> None:
        """Outside a Jupyter / IPython kernel, the function falls
        through to ``plt.show()`` (the legacy behaviour). The test
        verifies the function doesn't crash even on the Agg backend
        when no kernel is present.
        """
        import sys
        # Ensure no __IPYTHON__ is leaking.
        builtins_dict = sys.modules["builtins"].__dict__
        had_ipython = "__IPYTHON__" in builtins_dict
        if had_ipython:
            ipython_value = builtins_dict["__IPYTHON__"]
            del builtins_dict["__IPYTHON__"]
        try:
            # ``ps1`` absent -> non-interactive script. Short-circuit
            # path fires; no plot rendered. Function returns the DF.
            monkeypatch.delattr(sys, "ps1", raising=False)
            fi = np.array([0.5, 0.3, 0.2])
            columns = ["x", "y", "z"]
            df = plot_feature_importance(
                feature_importances=fi, columns=columns,
                kind="HEADLESS", show_plots=True, plot_file="",
            )
            assert df is not None
            assert "fi" in df.columns
        finally:
            if had_ipython:
                builtins_dict["__IPYTHON__"] = ipython_value


# ----------------------------------------------------------------------
# Renderer cleanup: matplotlib.use("Agg") removed
# ----------------------------------------------------------------------


class TestRendererDoesNotMutateBackend:
    """Lock-in: ``reporting/renderers/matplotlib.py`` must NOT call
    ``matplotlib.use(...)`` (this was the production root cause of FI
    plots not showing in Jupyter -- once Agg got locked in globally,
    downstream ``plt.show()`` calls were no-ops). The renderer creates
    its own ``FigureCanvasAgg(fig)`` explicitly; that's enough for the
    save path."""

    def test_renderer_source_does_not_call_matplotlib_use(self) -> None:
        from pathlib import Path
        src = Path(__file__).parent.parent / "reporting" / "renderers" / "matplotlib.py"
        if not src.exists():
            pytest.skip(f"renderer not found at {src}")
        text = src.read_text()
        # Allow ``matplotlib.use`` ONLY inside comments / docstrings.
        # The simplest lock: there must be NO ``matplotlib.use(`` call
        # outside a string / comment context. Crude but effective: count
        # uncommented occurrences.
        lines = text.splitlines()
        offending = [
            (i + 1, line)
            for i, line in enumerate(lines)
            if "matplotlib.use(" in line
            and not line.lstrip().startswith("#")
        ]
        assert not offending, (
            "regression: ``matplotlib.use(...)`` re-introduced in "
            f"renderer at lines {[i for i, _ in offending]}; this "
            "breaks Jupyter inline FI display. Remove the global "
            "backend mutation and rely on the explicit "
            "``FigureCanvasAgg(fig)`` already in the renderer."
        )
