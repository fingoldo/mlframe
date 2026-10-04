"""``generate_probs_from_outcomes`` must not index the per-bin offsets with the garbage ``int(NaN * nbins)`` when a chunk holds a NaN outcome.

Without bounds checking the stray index reads arbitrary memory and the NaN still propagates, so the defect is only visible in a process compiled with
``NUMBA_BOUNDSCHECK=1``; the test therefore runs there, in a fresh interpreter with its own cache directory.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np

_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from mlframe.calibration.probabilities import generate_probs_from_outcomes
    out = generate_probs_from_outcomes(np.array([0.0, np.nan, 1.0, 1.0, 0.0]), chunk_size=2, random_state=0, flip_percent=0.0)
    print(",".join(repr(float(v)) for v in out))
    """
)


def test_nan_outcome_chunk_is_poisoned_without_out_of_bounds_index(tmp_path):
    """Under bounds checking a NaN outcome poisons exactly its own chunk's probabilities and raises no IndexError."""
    env = dict(os.environ, NUMBA_BOUNDSCHECK="1", NUMBA_CACHE_DIR=str(tmp_path), NUMBA_NUM_THREADS="1")
    run = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, env=env, timeout=600)
    assert run.returncode == 0, run.stderr[-1500:]
    values = [float(v) for v in run.stdout.strip().splitlines()[-1].split(",")]
    assert np.isnan(values[:2]).all()
    assert np.isfinite(values[2:]).all()
