"""Pre-training baseline subsystem: dummy baselines + baseline diagnostics.

Two cohesive families share this package:

- ``dummy`` -- the per-(target_type, target_name) dummy-baseline report
  (``compute_dummy_baselines``) plus its end-of-suite summary formatters
  and the ``BaselineReport`` schema.
- ``diagnostics`` -- the cheap per-target ``BaselineDiagnostics`` pass
  (ablation / init-score / quick-model) that runs before the dummy
  baselines and its report formatter.

The public surface is re-exported here so existing
``from mlframe.training.baselines import X`` import sites resolve from the
documented package path; the heavy compute / kernel / formatting helpers
live in the ``_dummy_*`` and ``_baseline_diagnostics_*`` submodules.
"""
from __future__ import annotations

from .dummy import (
    compute_dummy_baselines,
    format_suite_end_summary,
    format_unified_target_verdict_table,
    BaselineReport,
    SCHEMA_VERSION,
    _baseline_inputs_hash,
    _warmup_numba_kernels,
)
from .diagnostics import (
    BaselineDiagnostics,
    BaselineDiagnosticsReport,
    AblationEntry,
    InitScoreBaseline,
    format_baseline_diagnostics_report,
)

__all__ = [
    "compute_dummy_baselines",
    "format_suite_end_summary",
    "format_unified_target_verdict_table",
    "BaselineReport",
    "SCHEMA_VERSION",
    "BaselineDiagnostics",
    "BaselineDiagnosticsReport",
    "AblationEntry",
    "InitScoreBaseline",
    "format_baseline_diagnostics_report",
]
