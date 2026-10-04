"""Verify that the project's pre-commit configuration enforces the agreed-upon set of static checks (ruff, black, mypy scoped to calibration) without requiring contributors to read the YAML by hand.

The constraints encoded here mirror code-arch-standards.md item 8: ruff + black are blocking pre-commit hooks; mypy is scoped to ``src/mlframe/calibration/`` only (Wave 5 strict beachhead); and the project black line-length is 160 (CLAUDE.md repeated rule).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _hooks_by_repo(config_text: str) -> dict[str, list[dict]]:
    """``{repo url: [hook dicts]}`` of a parsed pre-commit config; a repo listed twice has its hooks concatenated."""
    out: dict[str, list[dict]] = {}
    for repo in yaml.safe_load(config_text)["repos"]:
        out.setdefault(repo["repo"], []).extend(repo["hooks"])
    return out


def _precommit_hooks() -> dict[str, list[dict]]:
    """The repository's own ``.pre-commit-config.yaml`` as ``{repo url: [hook dicts]}``."""
    return _hooks_by_repo((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))


def _hook_ids(hooks: dict[str, list[dict]], repo_url: str) -> list[str]:
    """Hook ids declared under ``repo_url`` (empty when the repo is absent)."""
    return [h["id"] for h in hooks.get(repo_url, [])]


def test_hook_parser_reports_declared_repos_and_hooks_and_nothing_for_an_absent_one():
    """The parser maps a synthetic config to its repos and ids; an undeclared repo yields no ids."""
    hooks = _hooks_by_repo("repos:\n  - repo: https://x/a\n    rev: v1\n    hooks:\n      - id: one\n      - id: two\n  - repo: https://x/a\n    hooks:\n      - id: three\n")
    assert _hook_ids(hooks, "https://x/a") == ["one", "two", "three"]
    assert _hook_ids(hooks, "https://x/missing") == []


def test_pre_commit_config_has_ruff_hook() -> None:
    """The ruff-pre-commit repo is declared and carries a hook with id ``ruff``."""
    assert "ruff" in _hook_ids(_precommit_hooks(), "https://github.com/astral-sh/ruff-pre-commit")


def test_pre_commit_config_has_black_hook() -> None:
    """The black repo is declared and carries a hook with id ``black``."""
    assert "black" in _hook_ids(_precommit_hooks(), "https://github.com/psf/black")


def test_pre_commit_config_has_mypy_scoped_to_calibration() -> None:
    """mypy must run in pre-commit only against the strict beachhead, never the whole repo: the ``files`` regex accepts calibration sources and
    rejects the rest of the package. Expanding it requires a separate review per the gradual-typing policy."""
    mypy_hooks = [h for h in _precommit_hooks().get("https://github.com/pre-commit/mirrors-mypy", []) if h["id"] == "mypy"]
    assert len(mypy_hooks) == 1, "the mirrors-mypy repo must declare exactly one mypy hook"
    files = re.compile(mypy_hooks[0]["files"])
    assert files.search("src/mlframe/calibration/post.py")
    assert not files.search("src/mlframe/training/core.py")
    assert not files.search("src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py")


def test_pyproject_black_line_length_160() -> None:
    """[tool.black] line-length must stay at 160 (repeated user-feedback rule, CLAUDE.md)."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["black"]["line-length"] == 160


if __name__ == "__main__":
    sys_exit_code = pytest.main([__file__, "-v", "--no-cov"])
    raise SystemExit(sys_exit_code)
