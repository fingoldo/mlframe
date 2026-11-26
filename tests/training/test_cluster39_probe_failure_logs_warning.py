"""Cluster 39 regression: a failing degenerate-init probe is logged at WARNING, not DEBUG (audit 2026-11-20, finding 39-F62).

When the probe raises, the degenerate-weight check is skipped. That is non-fatal, but it must be visible. The test drives the probe into
its failure branch and checks the record it emits.
"""

import logging

import pytest

pytest.importorskip("lightning")  # the neural package subclasses Lightning classes: shards without the neural extras skip this module

import torch

from mlframe.training.neural import _flat_generate_helpers as fg


class _BrokenModel:
    """A model whose module walk raises, so the probe takes its failure branch."""

    def named_modules(self):
        """Fail the way a malformed model would."""
        raise RuntimeError("probe boom")


def test_probe_failure_is_logged_at_warning_and_names_the_skipped_check(caplog):
    """The probe's failure is logged at WARNING, names the skipped check and carries the cause (39-F62)."""
    with caplog.at_level(logging.DEBUG, logger=fg.logger.name):
        fg._generate_mlp_every_fit_init_probe(_BrokenModel())

    records = [r for r in caplog.records if "degenerate-init probe failed" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "degenerate-weight check skipped" in records[0].getMessage()
    assert "probe boom" in records[0].getMessage()


def test_healthy_model_logs_no_probe_failure(caplog):
    """Control: a normally initialised layer passes the probe silently (39-F62)."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with caplog.at_level(logging.WARNING, logger=fg.logger.name):
        fg._generate_mlp_every_fit_init_probe(model)

    assert not [r for r in caplog.records if "degenerate-init probe failed" in r.getMessage()]
