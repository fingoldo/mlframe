"""Verify that the project's pre-commit configuration enforces the agreed-upon set of static checks (ruff, black, mypy scoped to calibration) without requiring contributors to read the YAML by hand.

The constraints encoded here mirror code-arch-standards.md item 8: ruff + black are blocking pre-commit hooks; mypy is scoped to ``src/mlframe/calibration/`` only (Wave 5 strict beachhead); and the project black line-length is 160 (CLAUDE.md repeated rule).
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_text(path: Path) -> str:
    """Helper that read text."""
    return path.read_text(encoding="utf-8")


def test_pre_commit_config_has_ruff_hook() -> None:
    """Pre commit config has ruff hook."""
    cfg = _read_text(REPO_ROOT / ".pre-commit-config.yaml")
    assert "astral-sh/ruff-pre-commit" in cfg, "ruff pre-commit hook missing from .pre-commit-config.yaml"
    assert "id: ruff" in cfg, "ruff hook id not declared"


def test_pre_commit_config_has_black_hook() -> None:
    """Pre commit config has black hook."""
    cfg = _read_text(REPO_ROOT / ".pre-commit-config.yaml")
    assert "psf/black" in cfg, "black pre-commit hook missing from .pre-commit-config.yaml"
    assert "id: black" in cfg, "black hook id not declared"


def test_pre_commit_config_has_mypy_scoped_to_calibration() -> None:
    """mypy must run in pre-commit only against the calibration strict beachhead, never the whole repo. Expanding the files= regex requires a separate review per the gradual-typing policy."""

    cfg = _read_text(REPO_ROOT / ".pre-commit-config.yaml")
    assert "mirrors-mypy" in cfg, "mypy pre-commit hook missing from .pre-commit-config.yaml"
    assert "id: mypy" in cfg, "mypy hook id not declared"
    assert "src/mlframe/calibration/" in cfg, "mypy hook is not scoped to src/mlframe/calibration/"


def test_pyproject_black_line_length_160() -> None:
    """[tool.black] line-length must stay at 160 (repeated user-feedback rule, CLAUDE.md)."""

    pp = _read_text(REPO_ROOT / "pyproject.toml")
    assert "[tool.black]" in pp, "pyproject.toml missing [tool.black] section"
    black_section = pp.split("[tool.black]", 1)[1].split("\n[", 1)[0]
    assert "line-length = 160" in black_section, "[tool.black] line-length must equal 160 (project standard)"


if __name__ == "__main__":
    sys_exit_code = pytest.main([__file__, "-v", "--no-cov"])
    raise SystemExit(sys_exit_code)
