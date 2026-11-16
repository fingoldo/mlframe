"""Concurrent calls to install_log_sanitizer install the record factory exactly once."""

import logging
import threading

from mlframe.utils.log_sanitize import install_log_sanitizer, uninstall_log_sanitizer


def test_concurrent_install_returns_true_exactly_once():
    """Eight threads race to install; only one call may report that it installed the sanitizer."""
    uninstall_log_sanitizer()
    results = []
    barrier = threading.Barrier(8)

    def worker():
        """Wait at the barrier, then try to install."""
        barrier.wait()
        results.append(install_log_sanitizer())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results.count(True) == 1
    finally:
        uninstall_log_sanitizer()
        assert logging.getLogRecordFactory() is not None
