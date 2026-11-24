"""Cluster 28 regression: a failed early-stopping strip is logged at WARNING, not DEBUG (audit 2026-11-20, finding 28-F56).

The failure leaves native early stopping wired alongside slice early stopping, which changes training behaviour, so it must be visible in a
normal run. Source-level pin on the log level of that site.
"""

from pathlib import Path

from mlframe.training import _trainer_train_and_evaluate_parts as tp

_SRC = Path(tp.__file__).resolve()


def test_es_strip_failure_is_logged_at_warning():
    """The set_params early-stopping failure is logged at WARNING (28-F56, source pin)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "logger.warning(  # behaviour-changing: native ES stays wired" in text
    assert 'logger.debug(\n                                    "%s does not expose set_params' not in text
