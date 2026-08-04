"""log_only_except wave 4: closes 24 findings across evaluation/reports.py (3, best-effort chart
overlay/log-format), training/honest_diagnostics.py (3, already escalated via the returned
status="skipped"/reason contract), training/_feature_importances.py (3, best-effort -- feature
importance degrades to None), training/core/_phase_composite_discovery.py (3),
training/core/_phase_dummy_baselines.py (3), training/core/_phase_train_one_target_schema.py (3),
training/pipeline/_pipeline_extensions.py (3), and training/reporting/_reporting_diagnostics.py
(3, one already escalated via charts["failed"].append(...), two best-effort) -- all genuinely
non-fatal graceful-degradation sites marked with the scanner's recognized "best-effort" rationale
comment.
"""

from __future__ import annotations

import inspect


def test_evaluation_reports_sites_marked_best_effort():
    """Every chart-overlay/log-format except handler in evaluation/reports.py carries a marker."""
    import mlframe.evaluation.reports as reports_mod

    src = inspect.getsource(reports_mod)
    assert src.count("# best-effort:") >= 3


def test_honest_diagnostics_sites_marked_best_effort():
    """Every status="skipped"/reason-escalated except handler carries a marker."""
    import mlframe.training.honest_diagnostics as hd

    src = inspect.getsource(hd)
    assert src.count("# best-effort: escalated via the returned status=") == 3


def test_feature_importances_sites_marked_best_effort():
    """Every feature-importance except handler carries a marker."""
    import mlframe.training._feature_importances as fi

    src = inspect.getsource(fi)
    assert src.count("# best-effort: feature importance is optional diagnostic output") == 3


def test_phase_composite_discovery_sites_marked_best_effort():
    """Every composite-discovery except handler carries a marker."""
    import mlframe.training.core._phase_composite_discovery as disc

    src = inspect.getsource(disc)
    assert src.count("# best-effort:") == 3


def test_phase_dummy_baselines_sites_marked_best_effort():
    """Every dummy-baseline except handler carries a marker."""
    import mlframe.training.core._phase_dummy_baselines as db

    src = inspect.getsource(db)
    assert src.count("# best-effort:") == 3


def test_phase_train_one_target_schema_sites_marked_best_effort():
    """Every per-target schema/diagnostic except handler carries a marker."""
    import mlframe.training.core._phase_train_one_target_schema as schema_mod

    src = inspect.getsource(schema_mod)
    assert src.count("# best-effort:") == 3


def test_pipeline_extensions_sites_marked_best_effort():
    """Every optional preprocessing-extension except handler carries a marker."""
    import mlframe.training.pipeline._pipeline_extensions as ext

    src = inspect.getsource(ext)
    assert src.count("# best-effort:") == 3


def test_reporting_diagnostics_sites_marked_best_effort():
    """Every learning-curve diagnostic/chart except handler carries a marker."""
    import mlframe.training.reporting._reporting_diagnostics as rd

    src = inspect.getsource(rd)
    assert src.count("# best-effort:") == 3
