"""Cluster 08 regression: importing the transformer package does not import its ~130 modules, and every public name still resolves (audit 2026-11-20, finding 08-F16)."""

import os
import subprocess
import sys

import mlframe
import mlframe.feature_engineering.transformer as transformer


def test_every_public_name_resolves():
    """Each name in __all__ is reachable as an attribute and is not None."""
    missing = [n for n in transformer.__all__ if getattr(transformer, n, None) is None]
    assert not missing


def test_fresh_import_loads_only_a_few_transformer_modules():
    """A fresh interpreter importing the package loads far fewer than the 125 transformer submodules."""
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(mlframe.__file__)))  # -I ignores PYTHONPATH, so the child is pointed at this checkout's src
    code = "import sys; sys.path.insert(0, sys.argv[1]); import mlframe.feature_engineering.transformer; print(sum(1 for m in sys.modules if m.startswith('mlframe.feature_engineering.transformer.')))"
    out = subprocess.run([sys.executable, "-I", "-c", code, src_root], capture_output=True, text=True, check=True).stdout.strip()
    assert int(out) < 40
