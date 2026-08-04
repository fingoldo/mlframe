"""log_only_except wave 5: closes 3 findings across calibration/_post_train_calibrators.py (1,
best-effort sidecar .meta.json write), feature_selection/structure_discovery.py (1, already
documented "returning partial report" -- best-effort marker added alongside), and
training/crash_reporting.py (2, both already escalated via the returned ok=False sentinel) --
all genuinely non-fatal graceful-degradation sites marked with the scanner's recognized
"best-effort"/"documented" rationale comment.
"""

from __future__ import annotations

import inspect


def test_post_train_calibrators_sidecar_site_marked_best_effort():
    """The optional .meta.json sidecar write's except handler carries a marker."""
    import mlframe.calibration._post_train_calibrators as calibrators

    src = inspect.getsource(calibrators)
    assert "# best-effort:" in src


def test_structure_discovery_site_marked_best_effort():
    """The EDA detector-failure except handler carries a marker."""
    import mlframe.feature_selection.structure_discovery as sd

    src = inspect.getsource(sd)
    assert "best-effort:" in src


def test_crash_reporting_sites_marked_documented_or_best_effort():
    """Both crash-reporting except handlers (already escalated via ok=False) carry a marker."""
    import mlframe.training.crash_reporting as cr

    src = inspect.getsource(cr)
    assert src.count("escalated via the returned ok=False sentinel below") == 2
