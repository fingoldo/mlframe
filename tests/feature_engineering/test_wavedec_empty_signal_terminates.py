"""An empty signal is rejected by the wavelet decomposition instead of hanging inside njit (HNG4-1).

The symmetric-reflect kernel cycles between -1 and 0 forever when the signal length is 0, and a loop inside njit cannot be interrupted.
The call runs in a child process with a timeout, so a regression shows up as a failed test rather than a hung run.
"""

import subprocess
import sys

import pytest

pytest.importorskip("numba")

_CODE = """
import numpy as np
from mlframe.feature_engineering.wavelet_dwt import wavedec, wavedec_batched

for call in (lambda: wavedec(np.array([]), "db2", 1), lambda: wavedec_batched(np.zeros((3, 0)), "db2", 1)):
    try:
        call()
    except ValueError:
        print("ValueError")
    else:
        print("returned")
"""


def test_empty_signal_raises_value_error_and_does_not_hang():
    """Both entry points raise ValueError for length-0 input well inside the timeout."""
    done = subprocess.run([sys.executable, "-P", "-c", _CODE], capture_output=True, text=True, timeout=240, check=False)
    assert done.stdout.split() == ["ValueError", "ValueError"], done.stdout + done.stderr
