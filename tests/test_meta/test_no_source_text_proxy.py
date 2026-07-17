"""Meta-linter: no NEW source-text-proxy assertions in test files.

A source-text proxy is `assert "literal" in <var>` where <var> is bound from a
`*.read_text()` call -- it asserts that prod CODE CONTAINS a string instead of
asserting RUNTIME BEHAVIOUR, so it passes even when the behaviour is broken.

The existing population is BASELINED (`_source_proxy_baseline.json`, keyed by
``relpath::literal``); this linter fails only on NEWLY-added sites. As sites are
converted to real behavioural checks, drop their key from the baseline so it
shrinks. Refresh after intentionally accepting new proxies (rare):

  pytest tests/test_meta/test_no_source_text_proxy.py --refresh-source-proxy-baseline

Legitimate read_text uses (LOC budgets, CHANGELOG cross-walk, meta-linters,
docstring/annotation scanners) are NOT flagged: the scanner only matches a
string-constant membership test against a read_text-derived name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import orjson
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _source_proxy_scan import find_source_proxy_sites

_REPO_TESTS = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_source_proxy_baseline.json"


def _refresh_requested() -> bool:
    return "--refresh-source-proxy-baseline" in sys.argv


def _build_current() -> set[str]:
    keys: set[str] = set()
    for p in _REPO_TESTS.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(_REPO_TESTS).as_posix()
        for _ln, lit in find_source_proxy_sites(p):
            keys.add(f"{rel}::{lit}")
    return keys


def test_no_new_source_text_proxy_assertions() -> None:
    current = _build_current()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"),
            encoding="utf-8",
        )
        pytest.skip(f"source-proxy baseline refreshed ({len(current)} sites)")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[source-proxy] {len(fixed)} proxy site(s) DRAINED (converted to behaviour). "
            f"Refresh baseline to lock in: pytest {Path(__file__).name} "
            f"--refresh-source-proxy-baseline\n"
        )

    if new:
        pytest.fail(
            f'{len(new)} NEW source-text-proxy assertion(s): `assert "x" in <read_text-var>` '
            f"asserts code STRUCTURE as a stand-in for BEHAVIOUR. Replace with a real runtime "
            f"check (call the function / spy the call site / assert the effect):\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
