"""Meta-test: a function that takes its own seed never seeds a callee with an int literal.

The check is the shared py-ci-shared gate ``hardcoded_seed_in_library`` (see its module docstring for the failure it prevents).
"""

from __future__ import annotations

from pathlib import Path

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Benchmark scripts and vendored code are not the library's own behaviour.
_EXCLUDE = ("__pycache__", "_benchmarks/", "_vendored/")


def test_no_hardcoded_seed_in_library():
    """No seeded library function discards the caller seed for an int literal."""
    from py_ci_shared import hardcoded_seed_in_library as gate

    gate.assert_hardcoded_seed_in_library(MLFRAME_DIR, exclude=_EXCLUDE, min_files=500)
