"""A module-level function with no production call site enforces nothing.

A guard, a validator, a cleanup hook that nothing calls looks exactly like one that does: it is
defined, it is tested, it appears in review. The only difference is that whatever it enforces is not
enforced, and that difference shows up as the bug it was written to prevent rather than as anything
pointing here. A test calling it is not a production call site, and neither is an `__all__` entry or
a doctest -- that is how this class of dead control hides.

Baselined rather than gated: this repo has a large existing set, and the number that matters is
whether a NEW one appears. The baseline can only shrink -- an entry that gains a caller is drained
on the next refresh.

`ignore` takes bare names for the shapes this check cannot judge and must not guess at: entry points
the interpreter or a framework calls, and the lazy-module protocol.
"""

from __future__ import annotations

from pathlib import Path

import pytest

py_ci_shared = pytest.importorskip("py_ci_shared", reason="py-ci-shared is a dev-only git dependency")

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "_uncalled_functions_baseline.json"

#: Called by the interpreter, a framework or the import system rather than by name.
#: `__getattr__`/`__dir__` are PEP 562 lazy-module hooks; `main` is a console entry point;
#: `upgrade`/`downgrade` are alembic's.
_INTERPRETER_INVOKED = {"__getattr__", "__dir__", "main", "upgrade", "downgrade"}


def _production_files() -> list[Path]:
    """Shipped modules only. Benchmarks keep deliberately-uncalled baselines to measure against."""
    return sorted(p for p in (REPO_ROOT / "src").rglob("*.py") if "_benchmarks" not in p.as_posix())


def test_no_new_function_without_a_caller(request):
    """Kills: a new guard or validator that nothing ever calls."""
    from py_ci_shared.uncalled_functions import assert_no_new_uncalled_function

    files = _production_files()
    assert len(files) > 1000, f"only {len(files)} production files found -- the scan lost its subject"

    assert_no_new_uncalled_function(files, REPO_ROOT, BASELINE, ignore=_INTERPRETER_INVOKED)
