"""E3 — meta-test that ``mlframe.__version__`` is reported consistently
across every source-of-truth.

Sources checked:
  * ``mlframe.__version__``           (top-level package re-export)
  * ``mlframe.version.__version__``   (the version.py constant)
  * ``[project].version`` in ``pyproject.toml`` if present

mlframe currently uses a flat ``version.py`` only (no ``pyproject.toml``).
The test still asserts the two sources match — and grows automatically
if/when ``pyproject.toml`` is added.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import mlframe
from mlframe import version as version_module

REPO_ROOT = Path(mlframe.__file__).resolve().parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _read_pyproject_version() -> str | None:
    if not PYPROJECT_PATH.exists():
        return None
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[project"):
            in_project = stripped.startswith("[project]")
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = False
            continue
        if not in_project:
            continue
        m = re.match(r'version\s*=\s*"([^"]+)"', stripped)
        if m:
            return m.group(1)
    return None


def test_version_consistent_across_sources():
    sources: dict[str, str] = {}
    if hasattr(mlframe, "__version__"):
        sources["mlframe.__version__"] = mlframe.__version__
    if hasattr(version_module, "__version__"):
        sources["mlframe.version.__version__"] = version_module.__version__
    pyproject_version = _read_pyproject_version()
    if pyproject_version is not None:
        sources["pyproject.toml::[project].version"] = pyproject_version

    assert sources, "no version sources found — module layout broken?"
    distinct = set(sources.values())
    if len(distinct) > 1:
        details = "\n  ".join(f"{k} = {v!r}" for k, v in sources.items())
        pytest.fail(f"mlframe version disagrees across {len(distinct)} source(s):\n  " + details)
