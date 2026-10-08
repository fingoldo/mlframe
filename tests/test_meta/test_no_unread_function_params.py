"""Meta-test: a function never silently ignores sample_weight, seed, n_jobs, verbose, timeout or device.

The check is the shared py-ci-shared gate ``unread_function_params`` (see its module docstring for the failure it prevents).
"""

from __future__ import annotations

from pathlib import Path

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Benchmark scripts and vendored code are not the library's own behaviour.
_EXCLUDE = ("__pycache__", "_benchmarks/", "_vendored/")


def test_no_unread_function_params():
    """Every sample_weight/seed/random_state/n_jobs/verbose/timeout/device parameter of a library function is read."""
    from py_ci_shared import unread_function_params as gate

    gate.assert_unread_function_params(MLFRAME_DIR, exclude=_EXCLUDE, min_files=500)
