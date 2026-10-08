"""Meta-test: a digest or cache-key builder never returns one constant from its except handler.

The check is the shared py-ci-shared gate ``constant_fallback_cache_key`` (see its module docstring for the failure it prevents).
"""

from __future__ import annotations

from pathlib import Path

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Benchmark scripts and vendored code are not the library's own behaviour.
_EXCLUDE = ("__pycache__", "_benchmarks/", "_vendored/")


def test_no_constant_fallback_cache_key():
    """No key builder collapses distinct failing inputs onto a single constant key."""
    from py_ci_shared import constant_fallback_cache_key as gate

    gate.assert_constant_fallback_cache_key(MLFRAME_DIR, exclude=_EXCLUDE, min_files=500)
