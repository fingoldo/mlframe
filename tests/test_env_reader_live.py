"""``env_reader`` returns the current value of an environment variable: a change made after the reader was built is seen at the next call."""

import os

from mlframe.utils.env_flags import env_reader


def test_reader_sees_set_change_and_unset(monkeypatch):
    """Unset gives None, a later set and a later change are seen, a later delete gives None again."""
    monkeypatch.delenv("MLFRAME_TEST_ENV_READER", raising=False)
    read = env_reader("MLFRAME_TEST_ENV_READER")
    assert read() is None
    monkeypatch.setenv("MLFRAME_TEST_ENV_READER", "1")
    assert read() == "1"
    os.environ["MLFRAME_TEST_ENV_READER"] = "off"
    assert read() == "off"
    monkeypatch.delenv("MLFRAME_TEST_ENV_READER")
    assert read() is None


def test_reader_follows_a_replaced_environ(monkeypatch):
    """When ``os.environ`` itself is replaced by another mapping the reader reads that one."""
    read = env_reader("MLFRAME_TEST_ENV_READER_2")
    monkeypatch.setattr(os, "environ", {"MLFRAME_TEST_ENV_READER_2": "x"})
    assert read() == "x"
