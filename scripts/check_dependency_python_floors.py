"""Fail if any declared dependency floor excludes the project's own minimum Python.

A lower bound like ``pillow>=12.2.0`` silently becomes an install-time wall when that release
requires a newer Python than ``requires-python`` promises: the resolver reports only the FIRST such
conflict, so fixing them one CI run at a time is a whack-a-mole loop. This checks every requirement
in one pass against PyPI and prints the whole set.

Marker-gated requirements are evaluated first, so a floor already split by ``python_version`` is
correctly skipped for the Pythons it does not apply to.

Usage::

    python scripts/check_dependency_python_floors.py [--python 3.9] [--pyproject pyproject.toml]

Exits 1 if any floor excludes the target Python.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import tomllib
import urllib.request
from typing import Optional

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

# Marker variables other than python_version are pinned to a mainstream Linux/CPython runner: this
# check is about Python-version reachability, and letting platform markers vary would silently drop
# requirements from the scan on whichever machine happens to run it.
_BASE_MARKER_ENV = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}


def _requirement_floor(req: Requirement) -> Optional[Version]:
    """Lowest version this requirement admits, or None when it declares no lower bound."""
    floor = None
    for spec in req.specifier:
        if spec.operator in (">=", "==", "~="):
            try:
                candidate = Version(spec.version)
            except InvalidVersion:
                continue
            if floor is None or candidate > floor:
                floor = candidate
    return floor


def _applies_on(req: Requirement, python_version: str, extra: str) -> bool:
    """Whether req is active on the given Python once its marker is evaluated."""
    if req.marker is None:
        return True
    env = dict(_BASE_MARKER_ENV)
    env["python_version"] = python_version
    env["python_full_version"] = f"{python_version}.0"
    env["extra"] = extra
    return bool(req.marker.evaluate(env))


def _release_requires_python(name: str, version: Version) -> Optional[str]:
    """The ``requires_python`` PyPI records for one release, or None if it is unknown."""
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=60) as fh:  # nosec B310 - fixed https PyPI host
            requires = json.load(fh)["info"].get("requires_python")
    except Exception:
        return None
    return str(requires) if requires else None


def collect_requirements(pyproject_path: str, python_version: str) -> list:
    """Every (name, floor, extra) active on python_version, across core deps and all extras."""
    with open(pyproject_path, "rb") as fh:
        data = tomllib.load(fh)
    project = data["project"]
    entries = [(text, "core") for text in (project.get("dependencies") or [])]
    for extra, texts in (project.get("optional-dependencies") or {}).items():
        entries.extend((text, extra) for text in texts)

    out = []
    for text, extra in entries:
        try:
            req = Requirement(text)
        except Exception:
            continue
        if not _applies_on(req, python_version, extra):
            continue
        floor = _requirement_floor(req)
        if floor is not None:
            out.append((req.name, floor, extra))
    return out


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--python", default="", help="target Python, e.g. 3.9; defaults to the requires-python floor")
    args = parser.parse_args()

    with open(args.pyproject, "rb") as fh:
        requires_python = tomllib.load(fh)["project"].get("requires-python", "")
    target = args.python or _lowest_supported(requires_python)
    if not target:
        print(f"cannot infer a target Python from requires-python={requires_python!r}; pass --python", file=sys.stderr)
        return 2

    reqs = collect_requirements(args.pyproject, target)
    print(f"checking {len(reqs)} requirements active on Python {target} (requires-python = {requires_python})")

    def check(item):
        """Return a finding tuple when this floor excludes the target Python."""
        name, floor, extra = item
        requires = _release_requires_python(name, floor)
        if requires and not _supports_line(requires, target):
            return (name, str(floor), requires, extra)
        return None

    findings = []
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        for found in pool.map(check, reqs):
            if found:
                findings.append(found)

    if not findings:
        print(f"OK: no dependency floor excludes Python {target}")
        return 0
    print(f"\n{len(findings)} floor(s) exclude Python {target}:")
    for name, floor, requires, extra in sorted(findings):
        print(f"  [{extra}] {name}>={floor} requires_python={requires}")
    print("\nEither gate the floor behind a python_version marker or raise requires-python.")
    return 1


def _supports_line(requires_python: str, minor_line: str) -> bool:
    """Whether a release is installable on ANY patch of the given 3.x line.

    Testing only ``X.Y.0`` misreads the common security-exclusion form ``!=3.9.0,!=3.9.1,>=3.9``,
    which drops two early patches while supporting the rest of the line.
    """
    spec = SpecifierSet(requires_python)
    return any(spec.contains(f"{minor_line}.{patch}") for patch in (0, 1, 2, 5, 13, 23))


def _lowest_supported(requires_python: str) -> str:
    """Lowest 3.x that satisfies requires-python, so the check targets the promised minimum."""
    if not requires_python:
        return ""
    spec = SpecifierSet(requires_python)
    for minor in range(7, 30):
        if spec.contains(f"3.{minor}.0"):
            return f"3.{minor}"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
