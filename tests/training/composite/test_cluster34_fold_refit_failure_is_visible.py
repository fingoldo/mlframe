"""Audit 2026-11-20 finding 34-F9: a per-fold refit that raises is logged at WARNING and the caller gets None.

The caller then scores the fold with the global params, which saw that fold's held-out rows. Before the fix the failure was logged at DEBUG.
"""

import logging

import numpy as np

from mlframe.training.composite.discovery import _fold_refit


class _RaisingFitTransform:
    """A transform whose domain accepts every row and whose fit always raises."""

    def domain_check(self, y, base):
        """Accept every row."""
        return np.ones(len(y), dtype=bool)

    def fit(self, y, base, groups=None):
        """Always fail, as an unstable per-fold solve would."""
        raise RuntimeError("solver diverged")


def test_refit_failure_returns_none_and_logs_warning(caplog):
    """The refit returns None and a WARNING names the exception class."""
    y = np.arange(50, dtype=float)
    base = np.arange(50, dtype=float)
    with caplog.at_level(logging.DEBUG, logger=_fold_refit.logger.name):
        out = _fold_refit.refit_transform_on_fold(_RaisingFitTransform(), y, base)
    assert out is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "per-fold fit failed" in r.getMessage()]
    assert warnings and "RuntimeError" in warnings[0].getMessage()
