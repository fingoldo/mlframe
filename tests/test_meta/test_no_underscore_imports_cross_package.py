"""Meta-test: no production code imports a sibling-package's underscore-prefixed module from a non-sibling location.

Generalises the existing ``test_no_production_underscore_imports.py`` (which scopes to ``mlframe.training.core._*``) to the whole package tree. Underscore-prefixed modules under any ``src/mlframe/<pkg>/`` directory are conventionally internal-only -- the public surface is whatever the package re-exports from its own ``__init__.py``. Sibling files inside the same package may import each other freely; out-of-package production code must NOT reach across packages into ``other_pkg._private``.

Why this matters: deep underscore-imports across packages couple two subsystems to each other's private implementation details and silently break when either side moves a helper. They also accumulate dead code (no public ``__all__`` covers them) and surprise reviewers who expect ``mlframe.<pkg>.<thing>`` to come from the documented surface.

This sensor walks every file under ``src/mlframe`` and flags ``from mlframe.<pkg_a>.<sub_path>._<module> import ...`` (or ``import mlframe.<pkg_a>.<sub_path>._<module>``) when the importing file lives outside the ``<pkg_a>/<sub_path>/`` sibling cluster. ``_benchmarks/`` and ``_profile_*`` files are treated as test-adjacent and exempt.

An ALLOWLIST pins the 11 pre-existing cross-package underscore imports surfaced when this sensor first ran (2026-05-25; see ``audit/critique_2026_05_24/FINAL_VERIFICATION.md`` Wave-10 backlog). The list is intentionally explicit -- adding a new entry requires a one-line PR update, which surfaces the cross-package coupling at review time. The right cleanup direction is to promote the imported underscore-module's symbol into the owning package's public ``__init__.py`` and switch callers to the public path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mlframe"

# Frozen baseline of pre-existing cross-package underscore imports as of 2026-05-25.
# Each entry is (importer_relpath_posix, imported_module). New entries MUST NOT be added
# without a corresponding cleanup ticket; the goal is monotone shrinkage as the Wave-10+
# cleanup lands. Promoting a symbol into a package's public ``__init__.py`` and switching
# the importer to the public path is the standard cleanup direction.
ALLOWLIST: set[tuple[str, str]] = {
    ("src/mlframe/feature_engineering/transformer/residual_stratified_distance.py", "mlframe.training._gpu_probe"),
    ("src/mlframe/feature_selection/importance.py", "mlframe.metrics._calibration_plot"),
    ("src/mlframe/feature_selection/filters/_cat_interactions_step.py", "mlframe.feature_engineering.transformer._utils"),
    ("src/mlframe/feature_selection/wrappers/_rfecv_fit.py", "mlframe.training._ram_helpers"),
    ("src/mlframe/metrics/_core_numba_warmup.py", "mlframe.feature_selection.filters._prewarm"),
    ("src/mlframe/models/_ensembling_score.py", "mlframe.training._format"),
    ("src/mlframe/training/_composite_transforms_nonlinear.py", "mlframe.feature_selection.filters._kernel_tuning"),
    ("src/mlframe/training/_eval_helpers.py", "mlframe.metrics._calibration_plot"),
    ("src/mlframe/training/_feature_selection_config.py", "mlframe.feature_selection.wrappers._rfecv"),
}


def _is_test_adjacent(path: Path) -> bool:
    parts = path.parts
    if "_benchmarks" in parts:
        return True
    name = path.name
    if name.startswith("_profile_") or name.startswith("_bench_"):
        return True
    return False


def _iter_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def _underscore_module_dir(module: str) -> Path | None:
    """Return the directory of the sibling cluster the underscore-module lives in, or None if the module does not match the underscore-cross-package pattern.

    For ``mlframe.training.core._phase_helpers`` returns ``src/mlframe/training/core``.
    For ``mlframe.feature_selection.wrappers._rfecv`` returns ``src/mlframe/feature_selection/wrappers``.
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


def _collect_offenders() -> set[tuple[str, str]]:
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
