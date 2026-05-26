"""Wave 10 LOC-budget meta-test.

Scans ``src/mlframe/`` for ``.py`` files exceeding 1000 lines of code. After
Waves 6 + 10 the project's monolith-split policy is enforced: no file should
exceed the 1k LOC ceiling. Future PRs that re-introduce a >1k file are flagged
in CI so the splitting work doesn't drift back.

The exempt list is empty by design; any added entry must come with a
justification in the PR description (e.g. ``feature_engineering/wavelet_dwt.py``
WIP). If a hot file legitimately needs the budget raised, prefer carving it via
the sibling re-export pattern (see mlframe/CLAUDE.md "Monolith split").
"""

from __future__ import annotations

from pathlib import Path

import pytest


LOC_LIMIT = 1000

# Empty by design -- every mlframe .py file MUST stay <= 1000 LOC after Wave 10.
# Add a path here only if there's a documented, time-boxed reason and the PR
# carries an explicit FIXME for the next carve wave.
LOC_BUDGET_EXEMPT: set[str] = set()


def _src_root() -> Path:
    here = Path(__file__).resolve()
    # tests/test_meta/test_no_file_over_1k_loc.py -> repo root -> src/mlframe
    return here.parents[2] / "src" / "mlframe"


def _scan_src_for_oversize() -> list[tuple[str, int]]:
    root = _src_root()
    if not root.is_dir():
        pytest.skip(f"src tree not found at {root}; running from installed wheel?")
    over: list[tuple[str, int]] = []
    for path in root.rglob("*.py"):
        try:
            n = sum(1 for _ in path.open("r", encoding="utf-8"))
        except OSError:
            continue
        rel = path.relative_to(root.parent.parent).as_posix()  # "src/mlframe/..."
        if rel in LOC_BUDGET_EXEMPT:
            continue
        if n > LOC_LIMIT:
            over.append((rel, n))
    return sorted(over, key=lambda t: -t[1])


def test_no_mlframe_file_exceeds_1k_loc():
    over = _scan_src_for_oversize()
    if over:
        lines = [f"  {n:5d} LOC  {p}" for p, n in over]
        raise AssertionError(
            f"{len(over)} mlframe .py file(s) exceed {LOC_LIMIT} LOC. "
            f"Carve via sibling re-export pattern (CLAUDE.md: 'Monolith split'). "
            f"Oversized files:\n" + "\n".join(lines)
        )
