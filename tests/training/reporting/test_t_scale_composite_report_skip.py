"""Sensor: composite-target (T-scale) reports are skipped by default.

Background: composite targets train on the T-scale residual (e.g.
``T = cbrt(y) - alpha * X``). Per-target MAE/RMSE/R2 computed on T
are NOT comparable to leaderboard / raw-target reports. The y-scale
wrap pass emits a y-scale chart for the SAME (composite, inner_model)
pair from ``_phase_composite_wrapping`` so the operator gets a chart
on the comparable scale -- but the misleading T-scale chart in the
per-model reporter must NOT render.

User asked 2026-05-26: skip T-scale chart + log entirely for
composite targets, leave only the y-scale source.
2026-05-27 follow-up: the y-scale chart is now emitted by the wrap
pass; the per-model T-scale chart stays suppressed.

Detection: ``MTRESID=`` substring in ``model_name`` (stamped by
``select_target`` when the target is composite).
"""

from __future__ import annotations

from pathlib import Path


class TestTScaleCompositeReportSkip:
    """Groups tests covering t scale composite report skip."""
    @staticmethod
    def _render(tmp_path: Path, model_name: str) -> Path:
        """Runs the regression reporter with a chart file requested; returns the chart path."""
        import numpy as np

        from mlframe.training.reporting._reporting_regression import report_regression_model_perf

        rng = np.random.default_rng(0)
        targets = rng.normal(size=200)
        preds = targets + rng.normal(scale=0.2, size=200)
        chart = tmp_path / "chart.png"
        report_regression_model_perf(
            targets=targets, columns=["x"], model_name=model_name, model=None, preds=preds,
            print_report=False, show_perf_chart=False, plot_file=str(chart), metrics={},
        )
        return chart

    def test_source_skip_path_present(self, tmp_path, monkeypatch, caplog) -> None:
        """A T-scale (``MTRESID``) report writes no chart and logs the skip; the
        MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS opt-out and a y-scale label both still render it."""
        import logging

        monkeypatch.delenv("MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS", raising=False)
        with caplog.at_level(logging.INFO):
            skipped = self._render(tmp_path / "t", "cb MTRESID=y-cbrt-x")
        assert not skipped.exists()
        assert any("T-scale chart skipped here" in r.getMessage() for r in caplog.records)

        assert self._render(tmp_path / "y", "cb MTTR=y-cbrt-x").exists()

        monkeypatch.setenv("MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS", "1")
        assert self._render(tmp_path / "kept", "cb MTRESID=y-cbrt-x").exists()

    def test_opt_in_env_var_keyword(self) -> None:
        """Opt-out env var name is stable and matches the prefix
        convention (MLFRAME_*) used by other operator overrides."""
        from mlframe.training.reporting import _reporting_regression as rep

        src = Path(rep.__file__).read_text(encoding="utf-8")
        assert src.count("MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS") >= 1
