"""Meta-test: every [tool.deptry.per_rule_ignores] entry must name a currently-declared dependency.

deptry's DEP001/DEP002/DEP003 ignores in pyproject.toml have no built-in expiry mechanism (found
2026-07-18 architecture review): an ignore entry added for a real dependency stays in the list
forever, even after that dependency is later removed from [project.dependencies]/
[project.optional-dependencies]/[build-system] -- the per_rule_ignores list only grows, never
self-prunes, and deptry itself has no "is this ignore still load-bearing" check.

This test closes that gap for DEP002 (unused declared dependency) specifically: every DEP002
entry names a package that, BY DEFINITION of what DEP002 means, should still be traceable to a
real declaration somewhere in [project.dependencies]/[project.optional-dependencies]/
[build-system.requires] -- an ignore for a package that's been removed from every declaration is
a dead ignore nobody would notice went stale otherwise.

DEP001 (import missing from declared deps) and DEP003 (transitive dependency imported directly)
are NOT checked here, for the opposite reason each:
- DEP001 entries name *import* names for packages deliberately NOT declared in this file at all
  (that's the whole point of a DEP001 ignore) or local test-helper modules -- there's no
  declared-dependency list to cross-check them against.
- DEP003 entries are, by definition, packages that are NEVER directly declared (they're pulled in
  transitively by an already-declared direct dependency) -- checking them against the declared-
  dependency list would flag every single one as "stale" incorrectly. Verifying a DEP003 entry is
  still a genuine transitive dependency would require resolving the real installed dependency
  graph (importlib.metadata / pip show), a heavier check needing a real environment rather than a
  static pyproject.toml parse -- out of scope for this fast meta-test.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - repo's own floor is py39, but this test file itself runs under whatever collects it
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]

# Known first-party / non-PyPI exceptions: not declared as a normal dependency, but the DEP002/
# DEP003 comment block explains exactly why each is ignored anyway (invoked as a CLI tool, a
# transitive-only floor, a private sibling repo not on PyPI, etc.) -- see pyproject.toml's own
# [tool.deptry.per_rule_ignores] comments for the per-entry rationale.
KNOWN_NON_DECLARED_EXCEPTIONS = {
    "py-ci-shared",  # invoked as a CLI/pre-commit-hook command, never imported
    "finance",  # private sibling repo (github.com/fingoldo/finance), never on PyPI
    "cupy",  # deptry detects the IMPORT name ("cupy"), but the declared PyPI package names are
    # CUDA-variant-specific ("cupy-cuda12x", "cupy-cuda11x") -- never literally "cupy", so this
    # ignore never matches a normal declared-name lookup even though the dependency is very real.
}


def _normalize(name: str) -> str:
    """Returns ``name.lower().replace('_', '-')``."""
    return name.lower().replace("_", "-")


def _declared_dependency_names() -> set[str]:
    """Every package name declared anywhere in pyproject.toml: [project.dependencies],
    every [project.optional-dependencies] group, and [build-system.requires]."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    names: set[str] = set()
    _spec_re = re.compile(r"^[A-Za-z0-9_.\-]+")

    def _add_all(specs):
        """Test helper: for spec in specs: m = _spec_re.match(spec) if m: names.a...."""
        for spec in specs:
            m = _spec_re.match(spec)
            if m:
                names.add(_normalize(m.group(0)))

    _add_all(data.get("project", {}).get("dependencies", []))
    for group_specs in data.get("project", {}).get("optional-dependencies", {}).values():
        _add_all(group_specs)
    _add_all(data.get("build-system", {}).get("requires", []))
    return names


def _deptry_dep002_entries() -> list[str]:
    """The DEP002 list from [tool.deptry.per_rule_ignores]."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["tool"]["deptry"]["per_rule_ignores"].get("DEP002", [])


def test_deptry_dep002_ignores_are_not_stale():
    """Every DEP002 per_rule_ignores entry must still resolve to a declared dependency (or a
    documented first-party/non-PyPI exception) -- otherwise it's a dead ignore nobody would
    notice went stale, per this file's module docstring."""
    declared = _declared_dependency_names()
    packages = _deptry_dep002_entries()

    stale = [
        f"DEP002: {pkg!r} (normalized {_normalize(pkg)!r}) is not declared anywhere in pyproject.toml and is not a KNOWN_NON_DECLARED_EXCEPTIONS entry"
        for pkg in packages
        if _normalize(pkg) not in declared and pkg not in KNOWN_NON_DECLARED_EXCEPTIONS
    ]

    assert not stale, (
        "Stale deptry DEP002 per_rule_ignores entries found (named package no longer declared anywhere):\n  "
        + "\n  ".join(stale)
        + "\nEither the dependency was genuinely removed (delete the ignore entry too) or it's a "
        "real first-party/non-PyPI exception (add it to KNOWN_NON_DECLARED_EXCEPTIONS with a comment)."
    )
