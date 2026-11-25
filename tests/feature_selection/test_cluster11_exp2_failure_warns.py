"""Cluster 11 regression: a failing operator in the exp2 study is reported at WARNING, not DEBUG (audit 2026-11-20, finding 11-F5).

The study drops a failing operator and carries on; the drop is now visible at the default log level, once per operator.
"""

from pathlib import Path

FILE = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fe_operator_factory" / "stat_study" / "exp2_general.py"


def test_failing_operator_is_logged_at_warning():
    """The exp2 study logs an operator failure at WARNING through the throttled logger (11-F5)."""
    src = FILE.read_text(encoding="utf-8")
    assert 'logger.debug("operator %s failed' not in src
    assert "log_throttle(logger" in src and "logging.WARNING" in src
