"""Cluster 39 regression: the showcase module imports matplotlib lazily (audit 2026-11-20, finding 39-F16).

``matplotlib.pyplot`` used to be imported at module load, so importing the extractors package pulled in matplotlib even when no plot is drawn.
The import now lives inside the function that plots.
"""

from pathlib import Path

from mlframe.training.extractors import _extractors_showcase as sc


def test_module_does_not_import_pyplot_at_load():
    """No top-level matplotlib import remains in the showcase module (39-F16)."""
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    top_level = [line for line in src.splitlines() if line.startswith("import matplotlib") or line.startswith("from matplotlib")]
    assert top_level == []


def test_plotting_function_still_imports_pyplot():
    """Control: the histogram path keeps its matplotlib import inside the function (39-F16)."""
    src = Path(sc.__file__).resolve().read_text(encoding="utf-8")
    assert "import matplotlib.pyplot as plt" in src
