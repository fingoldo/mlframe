"""Cluster 11 regression: a failing study operator is reported at WARNING, once per operator (audit 2026-11-20, finding 11-F5).

The loop skipped failing operators at DEBUG, so an operator that raised on every input disappeared without a trace. The message now goes through
the throttled WARNING helper keyed per operator.
"""

from pathlib import Path

from mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study import core


def test_operator_failure_is_a_throttled_warning():
    """The operator-failure message is emitted as a throttled WARNING, not DEBUG (11-F5)."""
    src = Path(core.__file__).resolve().read_text(encoding="utf-8")
    assert 'logger.debug("operator %s failed on the study inputs' not in src
    assert 'log_throttle(logger, f"stat_study_op_failed:{bn}", logging.WARNING' in src
