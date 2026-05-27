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
    def test_source_skip_path_present(self) -> None:
        """Source-grep sensor: skip block + MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS
        env-var opt-out are wired into report_regression_model_perf
        (which lives in _reporting_regression.py)."""
        from mlframe.training import _reporting_regression as rep
        src = Path(rep.__file__).read_text(encoding="utf-8")
        # Skip message body changed 2026-05-27 to point at the wrap-pass
        # y-scale chart explicitly; sensor accepts either phrasing.
        assert (
            "T-scale chart skipped here" in src
            or "T-scale report skipped (composite target)" in src
        )
        assert "MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS" in src
        # The skip must short-circuit BEFORE the chart figure block
        # builds. Signature: detection on ``MTRESID`` substring near
        # the chart-block start, return preds_arr, None early.
        idx_signature = src.index('"MTRESID" in _model_name_str')
        # ``return preds_arr, None`` after the log line marks the
        # early-return path.
        idx_return = src.index("return preds_arr, None", idx_signature)
        assert idx_return > idx_signature

    def test_opt_in_env_var_keyword(self) -> None:
        """Opt-out env var name is stable and matches the prefix
        convention (MLFRAME_*) used by other operator overrides."""
        from mlframe.training import _reporting_regression as rep
        src = Path(rep.__file__).read_text(encoding="utf-8")
        assert src.count("MLFRAME_KEEP_T_SCALE_COMPOSITE_REPORTS") >= 1
