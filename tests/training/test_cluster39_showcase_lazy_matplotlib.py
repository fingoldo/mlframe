"""Cluster 39 regression: the showcase module imports matplotlib lazily (audit 2026-11-20, finding 39-F16).

``matplotlib.pyplot`` used to be imported at module load, so importing the extractors package pulled in matplotlib even when no plot is drawn.
The import now lives inside the function that plots.
"""

import subprocess
import sys

from mlframe.training.extractors import _extractors_showcase as sc


def test_importing_the_showcase_module_does_not_load_pyplot():
    """A fresh interpreter that imports the showcase module has no ``matplotlib.pyplot`` loaded (39-F16)."""
    code = "import sys; from mlframe.training.extractors import _extractors_showcase; " "print('matplotlib.pyplot' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300)
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_plotting_path_loads_pyplot_on_demand(monkeypatch):
    """Control: calling the target-distribution plotter with no targets still reaches the lazy pyplot import (39-F16)."""
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    try:
        sc._showcase_target_distributions({}, False, 0, 10)
    except ImportError:
        # Blocking pyplot with a None entry makes the lazy import raise, which proves the plotter imports it at call time.
        return
    raise AssertionError("plotter did not attempt the lazy matplotlib.pyplot import")
