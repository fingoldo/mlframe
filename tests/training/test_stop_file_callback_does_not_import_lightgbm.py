"""Importing the stop-file callbacks only probes for lightgbm, it does not import it (tracker 2026-11-02 wave 16, IMP16-8)."""

import subprocess
import sys


def test_importing_the_module_leaves_lightgbm_unimported():
    """The availability test was a real import (0.25 s) reached by every consumer of training.core."""
    code = "import sys, importlib; m = importlib.import_module('mlframe.training.callbacks.stop_file'); print('lightgbm' in sys.modules, m._HAS_LIGHTGBM)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert done.returncode == 0, done.stderr[-400:]
    imported, available = done.stdout.split()[-2:]
    assert imported == "False"
    assert available == "True"
