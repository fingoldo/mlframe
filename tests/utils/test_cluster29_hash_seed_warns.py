"""Audit 2026-11-20 finding 29-F9: asking set_random_seed for a hash seed warns that it does not change this process.

PYTHONHASHSEED is read at interpreter start. The test requests the hash seed and checks the WARNING says it only reaches child processes.
"""

import logging

from mlframe.utils import misc


def test_hash_seed_request_warns_that_it_only_reaches_child_processes(caplog, monkeypatch):
    """set_hash_seed=True logs a WARNING that names PYTHONHASHSEED and the child-process limit."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with caplog.at_level(logging.WARNING):
        misc.set_random_seed(7, set_hash_seed=True)
    assert any("child processes" in r.getMessage() and "PYTHONHASHSEED" in r.getMessage() for r in caplog.records)
