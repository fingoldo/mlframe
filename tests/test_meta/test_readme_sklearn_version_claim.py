"""X_OSS_HYGIENE_PACKAGING-1 regression test: README.md and CONTRIBUTING.md must not claim scikit-learn
1.5 support -- the sklearn-matrix CI workflow tests 1.6+ (it force-installs each pinned minor regardless
of the pyproject.toml floor, so its own tested range is independent of that floor's exact value).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_floor_is_actually_1_6_or_higher():
    """Sanity: pyproject.toml's own scikit-learn floor must be >=1.6 (the premise this test enforces).

    The floor itself may sit above 1.6 (mlframe's own production code can require newer sklearn features
    than category-encoders' bare importability floor) -- this only pins the MINIMUM acceptable floor.
    """
    src = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        "scikit-learn>=1.6" in src or "scikit-learn>=1.7" in src or "scikit-learn>=1.8" in src
    ), "pyproject.toml's scikit-learn floor dropped below 1.6 -- update this test and the README/CONTRIBUTING claims together"


def test_readme_does_not_claim_sklearn_1_5_support():
    """README.md must not claim the sklearn-matrix CI tests scikit-learn starting at 1.5."""
    src = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "1.5 through 1.8" not in src
    assert "scikit-learn 1.6 through 1.8" in src


def test_contributing_does_not_claim_sklearn_1_5_support():
    """CONTRIBUTING.md must not claim the sklearn-matrix workflow tests scikit-learn 1.5."""
    src = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "scikit-learn 1.5, 1.6, 1.7, 1.8" not in src
    assert "scikit-learn 1.6, 1.7, 1.8" in src
