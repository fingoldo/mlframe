"""Cluster 22 regression: unknown attributes of mlframe.calibration no longer import the heavy submodules (audit 2026-11-20, finding 22-F30)."""

import os
import subprocess
import sys

import mlframe

_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(mlframe.__file__)))


def _run(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return its stdout."""
    return subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); " + code, _SRC_ROOT], capture_output=True, text=True, check=True).stdout.strip()


def test_hasattr_probe_for_an_unknown_name_imports_nothing_heavy():
    """After ``hasattr(calibration, 'nope')`` neither quality nor probabilities is in sys.modules."""
    out = _run("import sys, mlframe.calibration as c; r = hasattr(c, 'no_such_name'); print(r, 'mlframe.calibration.quality' in sys.modules, 'mlframe.calibration.probabilities' in sys.modules)")
    assert out == "False False False"


def test_public_names_of_both_submodules_still_resolve():
    """A quality symbol and a probabilities symbol resolve lazily, and each import is only of the owning module."""
    out = _run("import sys, mlframe.calibration as c; f = c.generate_similar_probs_logit_space; print(callable(f))")
    assert out == "True"
