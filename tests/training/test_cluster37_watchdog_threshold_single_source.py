"""Cluster 37 regression: the composite watchdog threshold has a single definition (audit 2026-11-20, finding 37-F33).

``_WATCHDOG_RELATIVE_THRESHOLD`` was defined twice, and the copy in the post-phase module was never read. The live definition lives in
``_composite_wrap_watchdog`` and carries the rationale comment. The post-phase module no longer defines it and still imports.
"""

from pathlib import Path

from mlframe.training.core import _composite_wrap_watchdog as watchdog
from mlframe.training.core import _phase_composite_post as post


def test_post_module_no_longer_defines_the_threshold():
    """The duplicate definition is gone from the post-phase module (37-F33)."""
    src = Path(post.__file__).resolve().read_text(encoding="utf-8")
    assert "_WATCHDOG_RELATIVE_THRESHOLD = 0.01" not in src


def test_live_threshold_is_the_single_source():
    """The watchdog module defines the threshold once with the expected value (37-F33)."""
    src = Path(watchdog.__file__).resolve().read_text(encoding="utf-8")
    assert src.count("_WATCHDOG_RELATIVE_THRESHOLD = 0.01") == 1
    assert watchdog._WATCHDOG_RELATIVE_THRESHOLD == 0.01
