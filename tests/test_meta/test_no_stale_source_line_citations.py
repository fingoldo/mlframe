"""Meta-test: no log message embeds a ``<file>:<line>`` citation of its own source location.

The swallow-logging convention once wrote the emitting site into the message itself::

    logger.debug("suppressed in _adaptive_nbins.py:753: %s", e)

Every such citation is a lie the moment anything above it moves - and something always does (a carve, an
import re-order, a comment sweep). A repo-wide audit found 183 of 185 citations pointing at the wrong line,
several off by hundreds of lines and a few past the end of their own file. The line number is also redundant:
``logging`` already records the true module, function and line via the record's own metadata, and
``exc_info=True`` carries the traceback.

This gate is SELF-VERIFYING rather than baselined: a citation is checked against the position of the line
that emits it, so a correct citation passes by construction and no human judgement (or allowlist) is needed.
Because every historical citation has been stripped, the gate is clean rather than a ratchet - the fix for a
failure is to delete the ``<file>:<line>`` fragment, keeping the message text.
"""

from __future__ import annotations

import re
from pathlib import Path

import mlframe

from tests.test_meta._shared_ast_cache import source_text

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "profiling", "explore")

# "suppressed in _mah.py:118: %s" and the same shape under any other verb.
_CITATION = re.compile(r"(?P<file>[A-Za-z_][\w.]*\.py):(?P<line>\d+)")
# Only lines that are actually emitting a message; a docstring may legitimately reference another file:line.
_EMITTER = re.compile(r"\b(?:logger|logging|_module_logger|log)\s*\.\s*(?:debug|info|warning|error|exception|critical)\s*\(")


def _build_offending_list() -> list[str]:
    """``["relpath:lineno -> cited file:line (actual line N)", ...]`` for every self-citation that is wrong."""
    out: list[str] = []
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        text = source_text(py)
        if text is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _EMITTER.search(line):
                continue
            for m in _CITATION.finditer(line):
                # Only a citation of THIS file is self-referential; pointing at another module is a
                # (still fragile, but different) cross-reference this gate deliberately leaves alone.
                if m.group("file") != py.name:
                    continue
                cited = int(m.group("line"))
                if cited != lineno:
                    out.append(f"{rel}:{lineno} cites '{m.group(0)}' but is itself on line {lineno}")
    return out


def test_no_log_message_cites_a_stale_source_line():
    """No logging call embeds a ``<own-file>:<line>`` citation that disagrees with its own position."""
    offenders = _build_offending_list()
    assert not offenders, (
        f"{len(offenders)} logging call(s) embed a stale self-referential source-line citation. "
        "The cited line drifts on every edit above it and logging already records the true location. "
        "Fix: delete the '<file>:<line>' fragment from the message, keeping the rest of the text "
        "(e.g. 'suppressed in _mah.py:118: %s' -> 'suppressed: %s').\n  " + "\n  ".join(sorted(offenders)[:40])
    )
