"""Meta-test: variance, skewness and kurtosis are never built from raw power sums that cancel on large-offset data.

The check is the shared py-ci-shared gate ``cancellation_prone_moments`` (see its module docstring for the failure it prevents).
"""

from __future__ import annotations

from pathlib import Path

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Benchmark scripts and vendored code are not the library's own behaviour.
_EXCLUDE = ("__pycache__", "_benchmarks/", "_vendored/")


def test_no_cancellation_prone_moments():
    """No library function computes a moment as raw power sum minus squared mean; centre first or use Welford."""
    from py_ci_shared import cancellation_prone_moments as gate

    gate.assert_cancellation_prone_moments(MLFRAME_DIR, exclude=_EXCLUDE, min_files=500)
