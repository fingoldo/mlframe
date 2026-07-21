"""Meta-test that no production .py file contains mojibake (UTF-8 bytes misread as
CP1251/similar and re-saved) -- see F7 in audits/full_audit_2026-07-21/training_targets.md
and the same corruption independently found in a 4th file
(feature_engineering/_numerical_numba.py) during that fix's own repo-wide grep.

Detection principle: genuine mojibake of this kind is round-trip DETECTABLE. The corrupted
text is itself valid Unicode (e.g. Cyrillic-block characters), so ``run.encode("cp1251")``
succeeds; because those bytes are the ORIGINAL character's real UTF-8 encoding misread one
byte at a time, decoding them back as UTF-8 (``.decode("utf-8")``) ALSO succeeds and recovers
legible text different from the input. This double round-trip succeeding is a strong,
low-false-positive signal:
- Deliberately-authored non-ASCII text (e.g. a genuine Russian comment) also survives
  ``.encode("cp1251")`` (CP1251 is a valid Cyrillic encoding) but its bytes essentially never
  form valid UTF-8 when decoded back -- ``.decode("utf-8")`` fails, so it is correctly NOT
  flagged.
- A single already-correct special character with no CP1251 mapping at all (e.g. ``≥``,
  U+2265) fails at the ``.encode("cp1251")`` step and is correctly NOT flagged.

Snapshot-style like ``test_no_unicode_in_console_output.py``: any pre-existing offender is
captured in a baseline so landing this test doesn't require a rescan of the whole tree; new
mojibake introduced after this test lands fails the build immediately.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import orjson
import pytest

import mlframe

from tests.test_meta._shared_ast_cache import source_text

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_mojibake_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__",)

# Maximal runs of consecutive non-ASCII characters shorter than this are skipped: a lone
# accented character (rare but not impossible in a name/URL) round-tripping by coincidence is
# far more plausible at length 1 than a multi-character run doing so.
_MIN_RUN_LENGTH = 2

_NON_ASCII_RUN_RE = re.compile(r"[^\x00-\x7F]+")


def _refresh_requested() -> bool:
    """True if ``--refresh-mojibake-baseline`` was passed on the pytest command line."""
    return "--refresh-mojibake-baseline" in sys.argv


def _is_roundtrip_mojibake(run: str) -> bool:
    """True if ``run`` round-trips cleanly through cp1251-encode -> utf-8-decode into DIFFERENT, non-empty text."""
    if len(run) < _MIN_RUN_LENGTH:
        return False
    try:
        recovered = run.encode("cp1251").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
    return bool(recovered) and recovered != run


def _build_offending_set() -> set[str]:
    """``{relpath:lineno}`` for every line containing a round-trip-detectable mojibake run."""
    out: set[str] = set()
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        text = source_text(py)
        if text is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for lineno, line in enumerate(text.split("\n"), start=1):
            for match in _NON_ASCII_RUN_RE.finditer(line):
                if _is_roundtrip_mojibake(match.group()):
                    out.add(f"{rel}:{lineno}")
                    break  # one flag per line is enough to locate it
    return out


def test_no_new_mojibake_in_source():
    """No new round-trip-detectable mojibake beyond the frozen baseline."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"),
            encoding="utf-8",
        )
        pytest.skip(f"mojibake baseline refreshed at {_BASELINE_PATH.name} ({len(current)} line(s) flagged)")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_mojibake_in_source] {len(fixed)} line(s) DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh baseline to lock in.\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new line(s) contain mojibake (UTF-8 misread as CP1251 and re-saved -- "
            f"a lossy clipboard/terminal round-trip is the usual cause). Re-save the file as "
            f"clean UTF-8, OR refresh the baseline if this is a false positive:\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
