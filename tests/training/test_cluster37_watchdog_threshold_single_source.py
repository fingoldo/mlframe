"""Cluster 37 regression: the composite watchdog threshold has a single definition (audit 2026-11-20, finding 37-F33).

``_WATCHDOG_RELATIVE_THRESHOLD`` was defined twice, and the copy in the post-phase module was never read. The live definition lives in
``_composite_wrap_watchdog`` and carries the rationale comment. The post-phase module no longer exposes it and still imports.
"""

from mlframe.training.core import _composite_wrap_watchdog as watchdog
from mlframe.training.core import _phase_composite_post as post


def test_post_module_no_longer_exposes_the_threshold():
    """The post-phase module has no threshold attribute of its own (37-F33)."""
    assert not hasattr(post, "_WATCHDOG_RELATIVE_THRESHOLD")


def test_live_threshold_is_the_expected_value():
    """The watchdog module's threshold is the documented 1% relative tolerance (37-F33)."""
    assert watchdog._WATCHDOG_RELATIVE_THRESHOLD == 0.01
