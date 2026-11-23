"""Cluster 14 regression: a failed thread pin is logged at WARNING, not debug (audit 2026-11-20, finding 14-F13)."""

import logging

from mlframe.feature_selection.wrappers import _helpers as h


class _Est:
    """Estimator whose thread parameter is accepted but set_params always fails."""

    def get_params(self):
        """Advertise a thread parameter so the pin is attempted."""
        return {"n_jobs": 4, "num_threads": 4}

    def set_params(self, **kw):
        """Refuse the pin."""
        raise RuntimeError("cannot pin threads")


def test_failed_thread_pin_is_a_warning(caplog):
    """A refused pin is visible at WARNING with the parameter names (14-F13)."""
    with caplog.at_level(logging.WARNING, logger=h.logger.name):
        h._pin_threads_to_one(_Est())
    assert any(r.levelno == logging.WARNING and "thread-pinning" in r.getMessage() for r in caplog.records)
