"""Meta-test: run pyutilz.dev.code_audit's generic AST/SQL scanners against
this repo's own source, baseline-driven per this directory's snapshot-style
meta-test convention (see test_no_bare_except.py / test_no_mutable_defaults.py).

Findings are baselined together (keyed by ``check::file:line``) so
pre-existing debt doesn't block adoption -- only a NEW finding fails the
test. Refresh with ``--refresh-code-audit-baseline`` after a deliberate
change, or add a narrow, commented exclusion in the ``exclude_dirs``
passed below for a confirmed false positive.

Complements (does not replace) this directory's existing hand-rolled
scanners (bare_except, mutable_defaults, console_unicode, etc.) -- those
predate pyutilz.dev.code_audit's centralized versions and check
overlapping but not identical patterns. See
``pyutilz/src/pyutilz/dev/code_audit/__init__.py`` for what this scanner
suite additionally catches (default-via-or traps, logged-but-not-escalated
excepts, SQL LIMIT-without-ORDER-BY, dead CLI flags, non-idempotent SQL
migrations, duplicate conditions/dict-keys, discarded coroutines). Uses
the shared harness in py_ci_shared.code_audit_meta, same as every other
consumer.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.code_audit_meta import assert_no_new_code_audit_findings

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_code_audit_baseline.json"

# Mirrors this repo's pytest addopts (--ignore=legacy --ignore=benchmarks
# --ignore=profiling).
_EXCLUDE_DIRS = frozenset({"legacy", "benchmarks", "profiling", "_benchmarks"})


def test_no_new_code_audit_findings():
    """Fail only on NEW pyutilz.dev.code_audit findings vs the committed baseline."""
    assert_no_new_code_audit_findings(
        root=MLFRAME_DIR,
        baseline_path=_BASELINE_PATH,
        exclude_dirs=_EXCLUDE_DIRS,
    )
