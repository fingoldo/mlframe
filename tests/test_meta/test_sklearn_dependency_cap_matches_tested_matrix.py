"""Regression: pyproject.toml's scikit-learn upper bound must stay in sync with the highest version
actually exercised by the sklearn-matrix CI workflow, so it can never silently drift open again.

sklearn 1.9 combined with whatever category-encoders version CI's resolver falls back to on some legs
breaks CatBoostEncoder/TargetEncoder's own ``__sklearn_tags__`` override chain
(``AttributeError: 'super' object has no attribute '__sklearn_tags__'``) -- an upstream incompatibility,
not an mlframe bug, but with no upper bound on ``pyproject.toml``'s dependency spec the resolver picked
it up non-deterministically across CI legs, producing a wide, misleading failure cascade (worker
crashes, unrelated-looking model errors) that looked like dozens of independent bugs before being
traced back to this one root cause. ``pyproject.toml`` was pinned to ``<1.9`` to match what
``.github/workflows/sklearn-matrix-ci.yml`` (the workflow that actually tests specific sklearn minors)
already covers -- 1.6.1 through 1.8.0.

This test parses BOTH files and asserts they still agree, rather than hardcoding "1.9" in two places
that could drift apart independently: bump the CI matrix to cover a new sklearn minor and this test
will demand the pyproject.toml cap move with it (and vice versa) instead of one silently lagging.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_UPPER_BOUND_RE = re.compile(r'"scikit-learn>=1\.\d+,<(\d+)\.(\d+)(?:;[^"]*)?"')
_MATRIX_VERSION_RE = re.compile(r'-\s*"(\d+)\.(\d+)\.\d+"\s*(?:#.*)?$')


def _pyproject_upper_bound() -> tuple[int, int]:
    """The ``(major, minor)`` scikit-learn upper bound pyproject.toml declares."""
    src = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = _UPPER_BOUND_RE.search(src)
    assert m is not None, (
        "pyproject.toml's scikit-learn dependency spec no longer matches "
        "'\"scikit-learn>=1.N,<X.Y\"' -- update this regex (and check test_readme_sklearn_version_claim.py "
        "too) alongside whatever changed the spec's shape."
    )
    return int(m.group(1)), int(m.group(2))


def _matrix_sklearn_versions() -> list[tuple[int, int]]:
    """Every ``(major, minor)`` scikit-learn version listed in the sklearn-matrix-ci.yml matrix (the
    ``sklearn-version:`` list entries AND any ``include:`` row's ``sklearn-version:`` value).
    """
    src = (REPO_ROOT / ".github" / "workflows" / "sklearn-matrix-ci.yml").read_text(encoding="utf-8")
    versions: list[tuple[int, int]] = []
    for line in src.splitlines():
        stripped = line.strip()
        m = _MATRIX_VERSION_RE.match(stripped) or re.match(r'sklearn-version:\s*"(\d+)\.(\d+)\.\d+"\s*$', stripped)
        if m:
            versions.append((int(m.group(1)), int(m.group(2))))
    return versions


def test_matrix_sklearn_versions_parses_something():
    """Sanity: the matrix-version parser found at least the 3 primary sklearn-version entries.

    Guards against the regex silently matching nothing after an unrelated YAML reformat -- a test that
    can't observe its own premise (an empty list) would pass regardless of what the file actually says.
    """
    versions = _matrix_sklearn_versions()
    assert len(versions) >= 3, f"expected at least 3 scikit-learn versions in the CI matrix, parsed: {versions}"


def test_pyproject_cap_excludes_exactly_one_minor_above_the_highest_tested():
    """pyproject.toml's ``<X.Y`` cap must equal (highest tested minor + 1) -- neither looser (lets an
    untested, possibly-broken minor resolve) nor tighter (blocks a minor CI already proves works)."""
    upper_major, upper_minor = _pyproject_upper_bound()
    tested = _matrix_sklearn_versions()
    highest_major, highest_minor = max(tested)

    assert upper_major == highest_major, (
        f"pyproject.toml caps scikit-learn at <{upper_major}.{upper_minor}, but the CI matrix's highest "
        f"tested major is {highest_major} -- these should be the same major version family."
    )
    assert upper_minor == highest_minor + 1, (
        f"pyproject.toml caps scikit-learn at <{upper_major}.{upper_minor}, but the CI matrix's highest "
        f"tested version is {highest_major}.{highest_minor} (so the cap should be "
        f"<{highest_major}.{highest_minor + 1}). Bump the CI matrix to cover a new minor before loosening "
        "the cap, or tighten the cap if a tested minor was dropped from the matrix."
    )
