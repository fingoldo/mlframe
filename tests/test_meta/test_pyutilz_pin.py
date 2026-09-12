"""pyutilz is pinned to one commit everywhere, and the installed pyutilz includes it.

pyutilz is not on PyPI, so CI clones it by commit SHA in every workflow that needs it, and pyproject.toml's install
note names one too. Those pins had drifted to two different commits across 9 workflow files: a bump that misses a
line leaves some CI jobs testing a different pyutilz from the one the rest claim. And the editable pyutilz checkout a
development machine uses had fallen 8 commits behind master, so code-audit baselines refreshed there were built with
scanners no pin used. Both checks are ``py_ci_shared.git_dependency_pins``.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.git_dependency_pins import assert_installed_includes_pin, assert_pins_agree

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pin_files() -> list[Path]:
    """Every file that can carry a pyutilz pin: the CI workflows and pyproject.toml."""
    return [*sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")), REPO_ROOT / "pyproject.toml"]


def test_every_pyutilz_pin_names_one_commit():
    """All pyutilz pins in the workflows and pyproject.toml name the same commit."""
    files = _pin_files()
    # Fewer than 5 files means REPO_ROOT stopped pointing at the checkout; fewer than 10 pins (16 today) means the
    # pin spellings changed and the check is reading nothing.
    assert len(files) >= 5, f"only {len(files)} workflow files found; Check REPO_ROOT ({REPO_ROOT})"
    assert_pins_agree(files, "pyutilz", root=REPO_ROOT, min_pins=10)


def test_the_installed_pyutilz_includes_the_pin():
    """The pyutilz this interpreter imports contains the pinned commit (skips when installed from a plain copy)."""
    sha = assert_pins_agree(_pin_files(), "pyutilz", root=REPO_ROOT, min_pins=10)
    assert_installed_includes_pin("pyutilz", sha)
