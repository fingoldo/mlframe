"""Meta-test: process/audit metadata must not accumulate in code comments.

Finding IDs, ``Wave N`` / ``loop iter N`` markers, bare date stamps and audit-directory names belong in git
history and the PR description, not in the source. They are also actively misleading over time: an audit ID
outlives the audit, and a date stamp on a guard tells the next reader nothing about WHY the guard exists.

This ships as a RATCHET rather than a clean gate. A sweep already removed the unambiguous marker forms from
``feature_selection`` and every stale ``suppressed in <file>:<line>`` self-citation repo-wide, but a large
tail remains where a date sits INSIDE a prose sentence and deleting it by regex would destroy the meaning.
Those are baselined; only NEW metadata fails. Drain the baseline by rewriting comments, then refresh it.

Refresh with::

    pytest tests/test_meta/test_no_audit_metadata_in_comments.py --refresh-audit-metadata-baseline
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

import orjson
import pytest

import mlframe

from tests.test_meta._shared_ast_cache import source_text

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_audit_metadata_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "profiling", "explore")

# A comment carrying one of these is a preserved MEASUREMENT record, not process narration: the
# REJECTED-is-not-DELETED rule requires its date to stay, so the line is skipped entirely.
_SANCTIONED = re.compile(r"bench-attempt-rejected|VERDICT|RESULT \(|REJECTED", re.I)

_BANNED = {
    # "FE_STEP_B-8 fix", "(GPU_INFRA_A-11 fix," -- an audit finding ID.
    "finding-id": re.compile(r"\b[A-Z][A-Z_0-9]{4,}-\d+\b(?=\s*(?:fix|,|\)|:))"),
    # The audit directory itself.
    "audit-dir": re.compile(r"\bmrmr_audit_20\d\d-\d\d-\d\d\b"),
    "wave-marker": re.compile(r"\bWave\s+\d+"),
    "loop-iter": re.compile(r"\bloop\s+iter\s+\d+\b"),
    "date-stamp": re.compile(r"\b20\d\d-\d\d-\d\d\b"),
}

# ``Layer NN`` is the FE stack's own architecture vocabulary, not an audit marker -- deliberately NOT banned.


def _comment_lines(text: str):
    """Yield ``(lineno, comment_text)`` for real ``#`` comments only.

    Tokenizing (rather than scanning raw lines) keeps a ``#`` inside a string literal from being mistaken
    for a comment. Docstrings are left to prose review; this gate targets the comment stream.
    """
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                yield tok.start[0], tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return


def _refresh_requested() -> bool:
    """True if ``--refresh-audit-metadata-baseline`` was passed on the pytest command line."""
    return "--refresh-audit-metadata-baseline" in sys.argv


def _build_offending_set() -> set[str]:
    """``{"relpath:lineno:kind", ...}`` for every banned metadata marker in a comment."""
    out: set[str] = set()
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        text = source_text(py)
        if text is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for lineno, comment in _comment_lines(text):
            if _SANCTIONED.search(comment):
                continue
            for kind, pat in _BANNED.items():
                if pat.search(comment):
                    out.add(f"{rel}:{lineno}:{kind}")
    return out


def test_no_new_audit_metadata_in_comments():
    """No comment gains a finding ID, wave/iter marker, audit-dir name or date stamp beyond the baseline."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"audit-metadata baseline written with {len(current)} entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = current - baseline
    drained = baseline - current
    if drained:
        print(f"DRAINED {len(drained)} audit-metadata comment(s); refresh the baseline to lock the win in.", file=sys.stderr)

    assert not added, (
        f"{len(added)} comment(s) gained audit/process metadata. Finding IDs, 'Wave N' / 'loop iter N' markers, "
        "audit-directory names and date stamps belong in git history and the PR description -- a comment should "
        "say WHY the code is the way it is, which outlives any of them. Strip the marker and keep the prose. "
        "A preserved measurement record (bench-attempt-rejected / VERDICT / RESULT) is exempt and keeps its date."
        "\n  " + "\n  ".join(sorted(added))
    )
