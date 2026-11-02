"""Importing the training package does not import torch (tracker 2026-11-02 wave 16, IMP16-2)."""

import subprocess
import sys


def test_torch_is_not_loaded_by_importing_training_core():
    """Two modules imported torch at module level (12.5 s on the audit machine) for every consumer, although only MLP fits use it."""
    code = "import sys; import mlframe.training.core; print('torch' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=1500)
    assert done.returncode == 0, done.stderr[-400:]
    assert done.stdout.split()[-1] == "False"
