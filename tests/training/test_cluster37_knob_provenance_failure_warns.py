"""Cluster 37 regression: a failure while recording knob provenance is logged at WARNING (audit 2026-11-20, finding 37-F45).

The recorder is non-fatal, but a swallowed failure used to appear only at DEBUG, so provenance went missing with no visible sign. It now logs a
WARNING and returns; training is not interrupted.
"""

import logging
from unittest.mock import MagicMock

from mlframe.training.core import _main_train_suite_target_distribution as tdist


def test_provenance_failure_warns_and_does_not_raise(caplog):
    """A metadata object that cannot be written to logs a WARNING and returns normally (37-F45)."""
    metadata = MagicMock()
    metadata.setdefault.side_effect = RuntimeError("boom")
    with caplog.at_level(logging.WARNING):
        tdist._record_knob_provenance(metadata, None, None)
    assert any("knob provenance not recorded" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
