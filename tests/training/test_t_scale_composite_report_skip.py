"""Sensor: composite-target (T-scale) reports are skipped by default.

Background: composite targets train on the T-scale residual (e.g.
``T = cbrt(y) - alpha * X``). Per-target MAE/RMSE/R2 computed on T
are NOT comparable to the leaderboard / raw-target reports. The
y-scale wrap pass emits a separate ``[CompositeTargetEstimator]
y-scale metrics`` log block which IS comparable.

User asked 2026-05-26: skip T-scale chart + log entirely for
composite targets, leave only the y-scale source.

Detection: ``MTRESID=`` substring in ``model_name`` (stamped by
``select_target`` when the target is composite).
"""
from __future__ import annotations

from pathlib import Path


class TestTScaleCompositeReportSkip:
    def test_source_skip_path_present(self) -> None:
        """Source-grep sensor: skip block + MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS
        env-var opt-out are wired into report_regression_model_perf."""
        from mlframe.training import _reporting as rep
        src = Path(rep.__file__).read_text(encoding="utf-8")
        assert "T-scale report skipped (composite target)" in src
        assert "MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS" in src
        # The skip must happen BEFORE the chart-rendering block, so the
        # signature detection (`MTRESID=`) lives near the top of the
        # report function, not after the metrics_str assembly.
        idx_signature = src.index('"MTRESID=" in str(model_name)')
        idx_chart_block = src.index("_reg_subplots_kwargs = dict(")
        assert idx_signature < idx_chart_block, (
            "T-scale skip must short-circuit BEFORE the matplotlib chart "
            "block builds (otherwise we still pay the chart cost)."
        )

    def test_opt_in_env_var_keyword(self) -> None:
        """Opt-out env var name is stable and matches the prefix
        convention (MLFRAME_*) used by other operator overrides."""
        from mlframe.training import _reporting as rep
        src = Path(rep.__file__).read_text(encoding="utf-8")
        # Single canonical spelling, prefixed MLFRAME_
        assert src.count("MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS") >= 1
