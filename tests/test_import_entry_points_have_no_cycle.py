"""Each package can be the first thing imported in a fresh interpreter (tracker 2026-11-02, import cycle through mlframe.metrics).

mlframe.metrics.regression imported the feature_selection package at module level; that package imports estimators, which import mlframe.metrics.core while it
is still initialising, so any program that reached mlframe.metrics (or mlframe.calibration.probabilities) before the estimators failed with a circular-import
ImportError, and only a lucky import order hid it.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["mlframe.metrics", "mlframe.metrics.core", "mlframe.calibration.probabilities", "mlframe.estimators", "mlframe.feature_selection"],
)
def test_the_module_imports_first_in_a_fresh_interpreter(module):
    """``python -c 'import <module>'`` exits 0, so no entry point depends on another package having been imported first."""
    done = subprocess.run([sys.executable, "-P", "-c", f"import {module}"], capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-1500:]
