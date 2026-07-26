"""Regression tests for mlframe.utils.log_throttle."""
from __future__ import annotations

import logging

from mlframe.utils.log_throttle import log_throttle


def test_log_throttle_caps_emissions_per_key(caplog):
    """After max_count emissions for a key, further calls are suppressed with one notice."""
    logger = logging.getLogger("mlframe.test_log_throttle")
    key = f"test_key_{id(object())}"
    with caplog.at_level(logging.WARNING, logger=logger.name):
        for i in range(10):
            log_throttle(logger, key, logging.WARNING, "occurrence %d", i, max_count=3)
    messages = [r.message for r in caplog.records]
    assert sum(1 for m in messages if m.startswith("occurrence")) == 3
    assert sum(1 for m in messages if "suppressed" in m) == 1


def test_log_throttle_keys_are_independent(caplog):
    """Distinct throttle keys maintain independent counters and don't share a cap."""
    logger = logging.getLogger("mlframe.test_log_throttle")
    key_a = f"test_key_a_{id(object())}"
    key_b = f"test_key_b_{id(object())}"
    with caplog.at_level(logging.WARNING, logger=logger.name):
        for _ in range(5):
            log_throttle(logger, key_a, logging.WARNING, "from a")
            log_throttle(logger, key_b, logging.WARNING, "from b")
    messages = [r.message for r in caplog.records]
    assert sum(1 for m in messages if m == "from a") == 5
    assert sum(1 for m in messages if m == "from b") == 5


def test_log_throttle_exc_info_true_keeps_traceback(caplog):
    """exc_info=True (the logger.exception replacement case) attaches the active traceback."""
    logger = logging.getLogger("mlframe.test_log_throttle")
    key = f"test_key_exc_{id(object())}"
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise ValueError("boom")
        except ValueError as exc:
            log_throttle(logger, key, logging.ERROR, "failed: %s", exc, exc_info=True)
    assert caplog.records[0].exc_info is not None
    assert "ValueError: boom" in caplog.text
