"""Audit 2026-11-20 findings 28-F42 and 28-F49: a failed probe or an unreadable feature-name list is logged at WARNING.

Both used to log at DEBUG, invisible at default level. The test forces each failure and checks a WARNING record appears.
"""

import logging

from mlframe.training import train_eval


class _Unreadable:
    """A model whose feature_names_ attribute raises when read."""

    @property
    def feature_names_(self):
        """Raise, as a broken estimator would."""
        raise RuntimeError("feature names unreadable")


def test_unreadable_feature_names_warn(caplog):
    """The saved-names helper logs a WARNING and returns without the names."""
    with caplog.at_level(logging.WARNING):
        names = train_eval._saved_feature_names(_Unreadable())
    assert names is None
    assert any(r.levelno == logging.WARNING and "feature_names_" in r.getMessage() for r in caplog.records)
