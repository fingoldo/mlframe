"""Meta-test — every ``TODO`` / ``FIXME`` / ``XXX`` / ``HACK`` comment in
production code (under ``mlframe/`` outside ``tests/``) must carry an
attribution: either an assignee in parens (``TODO(name): ...``) or an
explicit date marker (``TODO 2026-04-28 ...``).

Catches the failure mode where TODOs accumulate as anonymous wishlist
items nobody owns. Symptoms: a 2-year-old "TODO: handle the empty
list case" that the original author has long since forgotten about,
and that surfaced last week as a P0 outage.

Strict-mode tests fail on un-attributed markers.  Warning-mode (a
separate test) reports total counts on stderr — the curation prompt.

Whitelist ``_GRANDFATHERED`` covers attribution-free markers the
maintainer accepts (e.g. external upstream issues, legacy carve-outs).
Drain over time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import mlframe
from pyutilz.dev.meta_test_utils import (
    ATTRIBUTION_RE,
    scan_todo_markers,
)

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Markers we audit. Lowercased before matching.
_MARKERS = ("TODO", "FIXME", "XXX", "HACK")

# Grandfathered un-attributed markers — drain over time.  Each entry
# is "rel/path.py:lineno" — same shape as the failure message.
# Surfaced 2026-04-28 by the meta-test's first run; maintainer to
# attribute or resolve in follow-up.
# All 5 markers surfaced 2026-04-28 have been attributed in-place
# (added ``2026-04-28`` ISO date). The set is empty and ready to flag
# any future un-attributed marker.
_GRANDFATHERED: set[str] = set()


def _scan_markers() -> list[tuple[Path, int, str, str]]:
    """Wraps the shared scanner from ``pyutilz.dev.meta_test_utils`` and
    excludes mlframe's ``legacy/`` carve-out (kept for archival purposes,
    not actively maintained)."""
    return scan_todo_markers(MLFRAME_DIR, extra_excludes=("legacy",))


def test_every_todo_marker_has_attribution():
    """Strict — un-attributed TODO/FIXME/XXX/HACK comments fail the test
    unless explicitly grandfathered.
    """
    bare: list[str] = []
    for path, lineno, kw, line in _scan_markers():
        rel = path.relative_to(MLFRAME_DIR).as_posix()
        ident = f"{rel}:{lineno}"
        if ident in _GRANDFATHERED:
            continue
        if not ATTRIBUTION_RE.search(line):
            bare.append(f"{ident}  {line[:100]}")

    if bare:
        pytest.fail(
            f"{len(bare)} {'/'.join(_MARKERS)} comment(s) without attribution. "
            f"Add an assignee in parens (``TODO(name): ...``) or an ISO date "
            f"(``TODO 2026-04-28: ...``). To grandfather, list "
            f'"<path>:<lineno>" in _GRANDFATHERED:\n  ' + "\n  ".join(bare[:30]) + (f"\n  ... and {len(bare) - 30} more" if len(bare) > 30 else "")
        )


# Total marker budget across all production mlframe files. The strict attribution test polices
# marker QUALITY (every marker is owned); this ceiling polices marker VOLUME so attributed-but-
# unbounded TODO accumulation fails the meta-test instead of sliding by as a stderr note. Lower
# the budget as markers are drained; raise it only with a deliberate decision.
_MARKER_BUDGET = 20


def test_todo_marker_count_within_budget():
    """Real assert (not a warning): total TODO/FIXME/XXX/HACK markers must stay within the budget.
    Still prints the per-marker / per-file breakdown so the curation prompt stays visible."""
    counts: dict[str, int] = {kw: 0 for kw in _MARKERS}
    by_file: dict[str, int] = {}
    total = 0
    for path, _, kw, _ in _scan_markers():
        counts[kw] += 1
        rel = path.relative_to(MLFRAME_DIR).as_posix()
        by_file[rel] = by_file.get(rel, 0) + 1
        total += 1
    top = sorted(by_file.items(), key=lambda kv: -kv[1])[:5]
    summary = (
        f"{total} marker(s) across {len(by_file)} file(s); breakdown: "
        + ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        + ". Top files: "
        + ", ".join(f"{p} ({n})" for p, n in top)
    )
    sys.stderr.write(f"\n[test_todo_marker_count_within_budget] {summary}\n")
    assert total <= _MARKER_BUDGET, f"TODO-marker volume {total} exceeds budget {_MARKER_BUDGET}; drain markers or raise the budget deliberately. {summary}"
