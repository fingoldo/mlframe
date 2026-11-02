"""Importing the training package does not import ngboost (tracker 2026-11-02 wave 16, IMP16-3)."""

import subprocess
import sys


def test_ngboost_is_not_loaded_by_importing_training_core():
    """Four modules imported ngboost at module level (about 4 s with sympy) although only the NGB model builder needs it."""
    code = "import sys; import mlframe.training.core; print('ngboost' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=1500)
    assert done.returncode == 0, done.stderr[-400:]
    assert done.stdout.split()[-1] == "False"
