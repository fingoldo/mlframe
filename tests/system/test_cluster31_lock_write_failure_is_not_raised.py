"""Audit 2026-11-20 finding 31-F4: an unwritable lock location makes _create_lock return False, not raise.

The pid file was written outside any try, so an OSError from the cache directory escaped ensure, which --advisory promises never to fail.
The test points the lock at a directory that does not exist and checks the call returns False.
"""

from mlframe.system.kernel_tuning_cache import _ensure


def test_unwritable_lock_location_takes_no_lock_and_does_not_raise(tmp_path):
    """A lock path under a missing directory returns False with no exception."""
    lock = tmp_path / "missing" / ".ensure.lock"
    assert _ensure._create_lock(lock) is False
