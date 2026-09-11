"""X_CICD_DEPENDENCIES-2 (2026-08-05 audit): pyproject.toml's dev extra pinned ``ruff==0.16.0`` with a
comment claiming it matched .pre-commit-config.yaml's ``ruff-pre-commit`` rev, but all three
ruff-pre-commit hooks there were pinned to v0.15.22 -- the exact stale-pin-drift class the prior audit's
F1 finding already fixed once (see ``test_f1_no_stale_ruff_pin_remains`` in
``test_x_cicd_dependencies_fixes.py``), now recurred with a different version pair. Meta-test: extract
every ruff version pin from pyproject.toml and .pre-commit-config.yaml and assert they all agree, so a
future silent drift between any of these pins fails CI instead of only surfacing at diff-review time.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    """Read a repo-relative file as UTF-8 text."""
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_ruff_version_pins_agree_across_pyproject_and_precommit():
    """Every ruff==X.Y.Z pin in pyproject.toml and every ruff-pre-commit rev in .pre-commit-config.yaml
    must reference the exact same version."""
    pyproject_text = _read("pyproject.toml")
    precommit_text = _read(".pre-commit-config.yaml")

    pyproject_pins = re.findall(r'"ruff==([\d.]+)"', pyproject_text)
    assert pyproject_pins, "expected an exact ruff==X.Y.Z pin in pyproject.toml's dev extra"

    precommit_pins = []
    for m in re.finditer(r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v([\d.]+)", precommit_text):
        precommit_pins.append(m.group(1))
    assert precommit_pins, "expected at least one ruff-pre-commit rev in .pre-commit-config.yaml"

    all_versions = set(pyproject_pins) | set(precommit_pins)
    assert len(all_versions) == 1, (
        f"ruff version pins have drifted apart: pyproject.toml has {sorted(set(pyproject_pins))}, "
        f".pre-commit-config.yaml's ruff-pre-commit revs have {sorted(set(precommit_pins))} -- all must match"
    )


def test_installed_tool_versions_match_their_exact_pins():
    """Every tool the dev extra pins exactly must be that version in the interpreter running the hooks.

    The blocking pre-commit hooks override the upstream ruff hook with ``language: system`` +
    ``entry: python -m ruff``, so pre-commit never builds the pinned isolated environment -- it runs the
    ambient interpreter's ruff, and the ``rev:`` guarantees nothing about what actually executed. The
    2026-09-08 review measured that gap live: every config said 0.16.1 while the installed ruff was
    0.15.22, which is exactly the "passes locally, fails in CI" failure the pin exists to prevent. The
    test above compares text to text; this one compares text to the binary that produces the verdicts.
    """
    import importlib.metadata as md

    pins = dict(re.findall(r'^\s*"([a-z0-9-]+)==([\d.]+)"', _read("pyproject.toml"), re.M))
    assert pins, "expected exact tool pins in pyproject.toml's dev extra"

    mismatched = {}
    for name, pinned in sorted(pins.items()):
        try:
            installed = md.version(name)
        except md.PackageNotFoundError:
            # A pinned tool absent from this environment is 1.4's finding, not this one's: the hook that
            # needs it fails closed with ModuleNotFoundError rather than passing silently.
            continue
        if installed != pinned:
            mismatched[name] = (pinned, installed)

    assert not mismatched, "installed tool versions differ from their exact pins (reinstall the dev extra): " + ", ".join(
        f"{n}: pinned {p}, installed {i}" for n, (p, i) in sorted(mismatched.items())
    )
