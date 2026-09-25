"""The pre-commit mypy hook runs through py_ci_shared.mypy_gate with a file floor derived from the measured source count.

A bare `python -m mypy` passes on exit status alone, so a run that aborts inside a stub or silently narrows its scope looks
clean. The floor must sit between 85% and 100% of the files mypy checks (mypy_gate reported 1668 on 2026-09-25; the count
below is every `.py` under src/mlframe outside the excluded benchmark/profiling trees). Re-measure with
`python -m py_ci_shared.mypy_gate src/mlframe` when the tree moves out of the band.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_FLOOR = re.compile(r"py_ci_shared\.mypy_gate --min-files (\d+) src/mlframe")
_EXCLUDED = {"legacy", "_benchmarks", "benchmarks", "profiling"}


def test_precommit_mypy_runs_through_mypy_gate_with_a_floor_sized_from_the_tree():
    """The blocking mypy hook asserts completion and names a floor within 85-100% of the checked source files."""
    floors = [int(m) for m in _FLOOR.findall((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))]
    assert len(floors) == 1, f"expected one mypy_gate --min-files hook over src/mlframe, found {floors}"
    measured = sum(1 for p in (REPO_ROOT / "src" / "mlframe").rglob("*.py") if not _EXCLUDED & set(p.parts))
    assert 0.85 * measured <= floors[0] <= measured, f"--min-files {floors[0]} is not within 85-100% of {measured} source files; re-measure"
