"""The package's internal import graph gains no new cycle, and no cycle member fails to import when it is loaded first.

A circular import can lurk for years because Python resolves a cycle at module level as long as the offending name is
read lazily; then a refactor moves one read to the top level and ``ImportError: cannot import name X from partially
initialized module Y`` ships. ``py_ci_shared.import_cycles`` finds every strongly connected component of the top-level
import graph (imports inside functions, under ``TYPE_CHECKING`` and under ``__main__`` excluded) and simulates each
member being imported first, reporting a ``from Y import n`` that runs while ``Y`` has not bound ``n`` yet.

The cycles the tree already has are recorded in ``_import_cycles_baseline.json``, each with why it resolves (mostly
monolith splits whose parent binds what the sibling needs before re-exporting it at its bottom). A new cycle fails; a
broken one must leave the baseline. Refresh with ``PY_CI_SHARED_REFRESH=import-cycles`` and write each new note by hand.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.import_cycles import assert_no_import_cycles

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_BASELINE = Path(__file__).resolve().parent / "_import_cycles_baseline.json"


def test_no_import_cycles_in_package():
    """No new top-level import cycle and no import-order failure, beyond the reasoned baseline."""
    # About 2,650 files parse under src/mlframe; the floor fails a scan that lost its subject.
    assert_no_import_cycles(MLFRAME_DIR, package="mlframe", baseline_path=_BASELINE, min_files=1000)
