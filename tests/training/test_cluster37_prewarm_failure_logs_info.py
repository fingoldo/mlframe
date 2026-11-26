"""Cluster 37 regression: a failed numba warm-up is logged at INFO, not DEBUG (audit 2026-11-20, finding 37-F41).

The warm-up is optional, but when it fails the first fit pays the JIT cost; an operator should see that at INFO level, not only at DEBUG.
The test forces the warm-up to raise and checks the record it emits.
"""

import logging
from types import SimpleNamespace

from mlframe.training.baselines import shared
from mlframe.training.core import _phase_config_setup as setup


def _configs(feature_selection: bool):
    """Minimal config objects that make the warm-up run, with feature selection on or off."""
    fs = SimpleNamespace(mrmr=object() if feature_selection else None, rfecv=None)
    return fs, SimpleNamespace(enabled=False), ["cb"], None, SimpleNamespace()


def test_warmup_failure_is_logged_at_info_with_the_cause(monkeypatch, caplog):
    """A raising warm-up produces one INFO record that carries the cause (37-F41)."""

    def _raise(**_kwargs):
        """Stand in for a warm-up that fails at JIT time."""
        raise RuntimeError("jit boom")

    monkeypatch.setattr(shared, "warmup_numba_kernels", _raise)
    with caplog.at_level(logging.DEBUG, logger=setup.logger.name):
        setup._prewarm_numba_kernels(*_configs(feature_selection=True))

    records = [r for r in caplog.records if "numba kernel warm-up failed" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "jit boom" in records[0].getMessage()


def test_successful_warmup_logs_no_failure(monkeypatch, caplog):
    """Control: a warm-up that completes emits no failure record (37-F41)."""
    monkeypatch.setattr(shared, "warmup_numba_kernels", lambda **_kwargs: None)
    with caplog.at_level(logging.INFO, logger=setup.logger.name):
        setup._prewarm_numba_kernels(*_configs(feature_selection=True))

    assert not [r for r in caplog.records if "numba kernel warm-up failed" in r.getMessage()]
