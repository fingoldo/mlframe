"""Meta-test: code that swaps ``sys.stdout``/``sys.stderr`` restores them only while its own stream is still installed.

The check is the shared py-ci-shared gate ``standard_stream_restore`` (see its module docstring for the failure it
prevents). The runtime half lives in ``tests/conftest.py``: ``_restore_closed_standard_streams`` fails the test that
leaves a stream replaced, and resets a stream an earlier test left closed so one leak cannot error every later test on
the worker.
"""

from __future__ import annotations

from pathlib import Path

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Benchmark scripts silence a library's chatter around a timing loop and are not imported by the package.
_EXCLUDE = ("__pycache__", "/legacy/", "/profiling/", "/explore/", "/_benchmarks/")


def test_no_unconditional_stream_restore_in_the_library():
    """Every swap of sys.stdout/sys.stderr under src/ restores only while its own stream is still installed."""
    from py_ci_shared import standard_stream_restore as gate

    gate.assert_standard_streams_restored_only_if_still_ours([MLFRAME_DIR], exclude=_EXCLUDE, min_files=500)
