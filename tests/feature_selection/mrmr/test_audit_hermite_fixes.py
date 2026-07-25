"""Regression tests for the hermite-wavelet findings of the mrmr audit fix wave.

Pins that importing the _hermite_fe_mi sibling FIRST (before its parent hermite_fe finishes) no longer
raises ImportError from the partially-initialised parent, and that a real MI dispatch call succeeds on that
sibling-first interpreter (the @njit kernels find the parent-resident _quantile_bin_njit global bound).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import mlframe


def test_hermite_fe_mi_sibling_first_import_and_dispatch():
    """A fresh interpreter that imports _hermite_fe_mi before hermite_fe must import cleanly AND run a real
    plugin_mi_classif_dispatch call (the njit kernel must find _quantile_bin_njit bound)."""
    code = (
        "import numpy as np\n"
        "import mlframe.feature_selection.filters._hermite_fe_mi as m\n"
        "rng = np.random.default_rng(0)\n"
        "x = rng.normal(size=400)\n"
        "y = (x > 0).astype(np.int64)\n"
        "v = m.plugin_mi_classif_dispatch(x, y, n_bins=8)\n"
        "assert np.isfinite(v) and v >= 0.0, v\n"
        "print('OK', v)\n"
    )
    # Propagate the importable path explicitly: relying on the parent shell's PYTHONPATH makes the test pass
    # or fail depending on how pytest was invoked, which would hide exactly the import fault it guards.
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(mlframe.__file__).resolve().parents[1])}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"sibling-first import/dispatch failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
    assert "OK" in r.stdout
