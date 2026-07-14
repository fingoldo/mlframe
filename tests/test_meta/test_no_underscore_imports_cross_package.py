"""Meta-test: no production code imports a sibling-package's underscore-prefixed module from a non-sibling location.

Generalises the existing ``test_no_production_underscore_imports.py`` (which scopes to ``mlframe.training.core._*``) to the whole package tree. Underscore-prefixed modules under any ``src/mlframe/<pkg>/`` directory are conventionally internal-only -- the public surface is whatever the package re-exports from its own ``__init__.py``. Sibling files inside the same package may import each other freely; out-of-package production code must NOT reach across packages into ``other_pkg._private``.

Why this matters: deep underscore-imports across packages couple two subsystems to each other's private implementation details and silently break when either side moves a helper. They also accumulate dead code (no public ``__all__`` covers them) and surprise reviewers who expect ``mlframe.<pkg>.<thing>`` to come from the documented surface.

This sensor walks every file under ``src/mlframe`` and flags ``from mlframe.<pkg_a>.<sub_path>._<module> import ...`` (or ``import mlframe.<pkg_a>.<sub_path>._<module>``) when the importing file lives outside the ``<pkg_a>/<sub_path>/`` sibling cluster. ``_benchmarks/`` and ``_profile_*`` files are treated as test-adjacent and exempt.

The ALLOWLIST is empty: the 9 baseline entries surfaced when this sensor first ran (2026-05-25) were promoted into public re-exports in Wave 11b. New entries must NOT be added without first attempting promotion to the owning package's public ``__init__.py``; this list exists as a safety net for genuinely-untreatable cases only.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mlframe"

# ALLOWLIST is empty: the 9 baseline entries (2026-05-25 snapshot) were promoted into public re-exports in Wave 11b. Each entry is ``(importer_relpath_posix, imported_module)`` -- add a new tuple here ONLY when promotion is genuinely infeasible (e.g. a cycle that cannot be broken without an architectural refactor); the standard cleanup is to expose the symbol via the owning package's public ``__init__.py``.
ALLOWLIST: set[tuple[str, str]] = set()


def _is_test_adjacent(path: Path) -> bool:
    """True for benchmark/profiling files exempt from the cross-package underscore-import check."""
    parts = path.parts
    if "_benchmarks" in parts:
        return True
    name = path.name
    if name.startswith("_profile_") or name.startswith("_bench_"):
        return True
    return False


def _iter_imports(tree: ast.AST):
    """Yield each imported module's dotted name from ``import X`` and ``from X import ...`` statements."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def _underscore_module_dir(module: str) -> Path | None:
    """Return the directory of the sibling cluster the underscore-module lives in, or None if the module does not match the underscore-cross-package pattern.

    For ``mlframe.training.core._phase_helpers`` returns ``src/mlframe/training/core``.
    For ``mlframe.feature_selection.wrappers.rfecv._fit`` returns ``src/mlframe/feature_selection/wrappers/rfecv``.
    For ``mlframe.utils.safe_pickle`` (no underscore in any segment) returns None.
    """
    if not module.startswith("mlframe."):
        return None
    parts = module.split(".")
    # Find the first segment starting with "_" (after "mlframe").
    for idx, seg in enumerate(parts[1:], start=1):
        if seg.startswith("_"):
            # Sibling cluster directory = everything up to (but not including) the underscore segment.
            cluster_path = SRC.joinpath(*parts[1:idx])
            return cluster_path
    return None


def _file_in_cluster(py_path: Path, cluster_dir: Path) -> bool:
    """True if ``py_path`` lives inside ``cluster_dir`` (or its subdirectories)."""
    try:
        py_path.relative_to(cluster_dir)
        return True
    except ValueError:
        return False


@cache
def _collect_offenders() -> set[tuple[str, str]]:
    """Cached: both tests below call this to full-scan ``src/mlframe``; without caching
    the whole-tree ast.parse pass runs twice per pytest session for identical input."""
    offenders: set[tuple[str, str]] = set()
    for py_path in SRC.rglob("*.py"):
        if _is_test_adjacent(py_path):
            continue
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for mod in _iter_imports(tree):
            cluster_dir = _underscore_module_dir(mod)
            if cluster_dir is None:
                continue
            if not cluster_dir.exists():
                continue
            if _file_in_cluster(py_path, cluster_dir):
                continue
            offenders.add((py_path.relative_to(REPO_ROOT).as_posix(), mod))
    return offenders


def test_no_new_underscore_imports_cross_package() -> None:
    """No new cross-package underscore imports beyond the frozen 2026-05-25 baseline."""
    offenders = _collect_offenders()
    new = offenders - ALLOWLIST
    if new:
        formatted = "\n".join(f"  {p} -> {m}" for p, m in sorted(new))
        pytest.fail(
            "New cross-package underscore imports detected (not in ALLOWLIST). Production code must "
            "not import a sibling-package's underscore-prefixed module from a non-sibling location. "
            "Promote the symbol into the owning package's public ``__init__.py`` and switch the importer "
            "to the public path. New offenders:\n" + formatted
        )


def test_underscore_imports_allowlist_is_not_stale() -> None:
    """Allowlist must not list entries that have already been cleaned up."""
    offenders = _collect_offenders()
    stale = ALLOWLIST - offenders
    if stale:
        formatted = "\n".join(f"  {p} -> {m}" for p, m in sorted(stale))
        pytest.fail(
            "ALLOWLIST entries are stale (no longer present in source). Remove them from the allowlist "
            "to keep the baseline monotone-shrinking. Stale entries:\n" + formatted
        )
