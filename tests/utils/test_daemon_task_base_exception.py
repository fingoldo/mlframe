"""``start_daemon_task`` must settle its future for ``BaseException`` too, so a waiting caller never hangs."""

from mlframe.utils.daemon_task import start_daemon_task


def _raise(exc_type):
    """Raise ``exc_type`` with a fixed message."""
    raise exc_type("boom")


def test_base_exception_settles_future_instead_of_hanging():
    """A ``SystemExit`` from the task reaches the future, so ``exception()`` returns rather than blocking."""
    fut = start_daemon_task(_raise, SystemExit)
    exc = fut.exception(timeout=5)
    assert isinstance(exc, SystemExit)


def test_ordinary_exception_settles_future_with_same_instance():
    """An ordinary ``Exception`` is delivered through the future with its own type."""
    fut = start_daemon_task(_raise, ValueError)
    exc = fut.exception(timeout=5)
    assert isinstance(exc, ValueError)
