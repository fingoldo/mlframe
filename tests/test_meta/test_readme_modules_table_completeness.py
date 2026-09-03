"""X_OSS_HYGIENE_PACKAGING-3 regression test: README.md's Modules table must list every real,
substantial top-level mlframe subpackage.

Pre-fix, ``mlframe.competition`` (~3956 LOC) and ``mlframe.data_valuation`` (~1372 LOC) had no row in
the table despite being real, documented subpackages. Generalized beyond just those two names so future
subpackage additions get the same drift check, not just a one-off regex pin.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MLFRAME_SRC = REPO_ROOT / "src" / "mlframe"

# Internal/private subpackages that are never meant to be user-facing entries in the README's public
# module overview -- excluded so this test doesn't demand documentation for implementation details.
_EXCLUDED_SUBPACKAGES = {"_benchmarks", "__pycache__"}
_MIN_LOC_TO_REQUIRE_DOCUMENTATION = 200


def _subpackage_loc(pkg_dir: Path) -> int:
    """Total line count across every .py file in pkg_dir, recursively."""
    total = 0
    for py_file in pkg_dir.rglob("*.py"):
        try:
            total += len(py_file.read_text(encoding="utf-8", errors="ignore").splitlines())
        except OSError:
            continue
    return total


def test_readme_modules_table_lists_every_substantial_subpackage():
    """Every top-level mlframe subpackage over the LOC threshold must appear as `mlframe.<name>` in README.md."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    missing = []
    for entry in sorted(MLFRAME_SRC.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_") or entry.name in _EXCLUDED_SUBPACKAGES:
            continue
        if not (entry / "__init__.py").exists():
            continue
        if _subpackage_loc(entry) < _MIN_LOC_TO_REQUIRE_DOCUMENTATION:
            continue
        if f"mlframe.{entry.name}" not in readme:
            missing.append(entry.name)
    assert not missing, f"README.md's Modules table is missing substantial subpackage(s): {missing}"


def test_readme_lists_competition_and_data_valuation():
    """Sanity pin for the two specific packages this finding named."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "mlframe.competition" in readme
    assert "mlframe.data_valuation" in readme
