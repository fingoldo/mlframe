"""Regression guard for the module-level import cycles the code-audit ``import_cycle`` scanner reports.

Each pair below is a split-module family where the child used to import its parent at module top-level while the
parent re-exported the child from its bottom. The child's import now happens inside the function that needs it, or
the shared name lives in a leaf module both sides import from. These tests fail if a top-level edge back to the
parent is reintroduced.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# (child module file relative to mlframe, parent module dotted name it must not import at top level)
_CHILD_PARENT_EDGES = [
    ("training/pipeline/_pipeline_helpers_apply.py", "mlframe.training.pipeline._pipeline_helpers"),
    ("training/core/_phase_helpers_fit_pipeline.py", "mlframe.training.core._phase_helpers_fit_split"),
    ("training/composite/discovery/_screening_tiny_perbin.py", "mlframe.training.composite.discovery._screening_tiny"),
    ("metrics/classification/_classification_extras_blocks.py", "mlframe.metrics.classification._classification_extras"),
    ("training/reporting/_reporting_diagnostics.py", "mlframe.training.reporting._reporting"),
    ("training/reporting/_reporting_probabilistic.py", "mlframe.training.reporting._reporting"),
    ("training/reporting/_reporting_probabilistic_helpers.py", "mlframe.training.reporting._reporting"),
    ("training/reporting/_reporting_probabilistic_calib.py", "mlframe.training.reporting._reporting_probabilistic"),
    ("training/reporting/_reporting_regression/__init__.py", "mlframe.training.reporting._reporting"),
    ("training/targets/_target_distribution_analyzer_features.py", "mlframe.training.targets._target_distribution_analyzer"),
    ("training/targets/_target_distribution_analyzer_modes.py", "mlframe.training.targets._target_distribution_analyzer"),
    ("training/targets/_target_distribution_analyzer_target_fn.py", "mlframe.training.targets._target_distribution_analyzer"),
    ("training/targets/_target_temporal_audit_from_agg.py", "mlframe.training.targets.target_temporal_audit"),
    ("training/targets/_target_temporal_changepoint.py", "mlframe.training.targets.target_temporal_audit"),
    ("feature_selection/filters/_gpu_resident_discretize.py", "mlframe.feature_selection.filters._gpu_resident_fe"),
    ("feature_selection/filters/_gpu_resident_k_chunk_ktc.py", "mlframe.feature_selection.filters._gpu_resident_fe"),
    ("feature_selection/filters/_gpu_resident_materialise.py", "mlframe.feature_selection.filters._gpu_resident_fe"),
    ("feature_selection/filters/_gpu_resident_pair_mi.py", "mlframe.feature_selection.filters._gpu_resident_fe"),
    ("feature_selection/filters/_feature_engineering_pairs/_pairs_chunks.py", "mlframe.feature_selection.filters.feature_engineering"),
]


def _top_level_imported_modules(path: Path, dotted: str) -> set[str]:
    """Absolute dotted names bound at import time: module body and ``if`` blocks, except ``TYPE_CHECKING`` bodies; defs excluded."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg_parts = dotted.split(".")[:-1]
    out: set[str] = set()
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                stack.extend(node.orelse)
                continue
            stack.extend(node.body)
            stack.extend(node.orelse)
            continue
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)] if node.level > 1 else pkg_parts
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            out.add(target)
            out.update(f"{target}.{a.name}" for a in node.names)
    return out


@pytest.mark.parametrize("child_rel, parent_dotted", _CHILD_PARENT_EDGES, ids=[c for c, _ in _CHILD_PARENT_EDGES])
def test_child_does_not_import_parent_at_module_top(child_rel, parent_dotted):
    """The child module must not bind its parent at import time."""
    child = MLFRAME_DIR / child_rel
    child_dotted = "mlframe." + child_rel[:-3].replace("/", ".")
    if child.name == "__init__.py":
        child_dotted = "mlframe." + child_rel[: -len("/__init__.py")].replace("/", ".") + ".__init__"
    imported = _top_level_imported_modules(child, child_dotted)
    assert parent_dotted not in imported, f"{child_rel} imports {parent_dotted} at module top-level again: this closes an import cycle"
