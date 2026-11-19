"""SHAP values of tree models are computed by mlframe's own TreeSHAP (``mlframe.inference.tree_shap``), not by a direct call into the public ``shap`` library.

The public library is allowed to draw plots and to answer the families we do not implement, and the places that do that are listed below with the reason. Any other source file that creates
``shap.TreeExplainer(...)``, ``shap.Explainer(...)`` or another ``shap.*Explainer`` (or a bare ``TreeExplainer(...)`` without importing ours) fails this test: route it through
``mlframe.inference.tree_shap``. The scan walks the syntax tree, so a call in a comment, a docstring or a string is not a call.
"""

from __future__ import annotations
from tests.test_meta._scan_guard import assert_scanned_enough

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"

# file (relative to src/mlframe) -> why a direct call into the public library stays
ALLOWED = {
    "inference/tree_shap.py": "the fallback for the families mlframe does not implement (multiclass, sklearn forests) and the plotting wrapper",
    "feature_selection/shap_proxied_fs/_shap_proxy_explain.py": "the logged fallback of _shap_phi_and_base for a model the table kernels do not cover",
    "feature_selection/shap_proxied_fs/_shap_proxy_interactions.py": "the fallback of compute_interaction_tensor (unsupported families)",
    "reporting/charts/shap_panels.py": "the shim path for the families mlframe does not implement, and shap.KernelExplainer for non-tree models (no tree implementation exists for them)",
    "feature_selection/wrappers/_importance_methods.py": "shap.Explainer for non-tree models (linear, permutation): not tree SHAP, nothing to reimplement",
    "inference/explainability.py": "shap.Explainer for non-tree model stubs; the tree models take mlframe's TreeSHAP",
}

_EXPLAINERS = {"TreeExplainer", "Explainer", "KernelExplainer", "LinearExplainer", "DeepExplainer", "GPUTreeExplainer", "PermutationExplainer", "SamplingExplainer"}
_PUBLIC_NAMES = {"shap", "_shap"}
_OURS = "mlframe.inference.tree_shap"


def _direct_calls(path: Path):
    """``(line, text)`` of every call that creates an explainer of the public library in ``path``: ``shap.X(...)`` / ``_shap.X(...)``, or a bare ``TreeExplainer(...)`` when the file does not import ours."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports_ours = any(isinstance(n, ast.ImportFrom) and n.module == _OURS for n in ast.walk(tree)) or any(
        isinstance(n, ast.Import) and any(a.name == _OURS for a in n.names) for n in ast.walk(tree)
    )
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _EXPLAINERS and isinstance(func.value, ast.Name) and func.value.id in _PUBLIC_NAMES:
            found.append((node.lineno, f"{func.value.id}.{func.attr}("))
        elif isinstance(func, ast.Name) and func.id == "TreeExplainer" and not imports_ours:
            found.append((node.lineno, "TreeExplainer("))
    return found


def test_no_direct_public_shap_computation_outside_the_listed_places():
    """Scan src/mlframe (benchmarks excluded: they exist to compare against the public library) for explainers of the public library."""
    offenders = []
    files = sorted(SRC.rglob("*.py"))
    assert_scanned_enough(len(files), "src/mlframe")
    for path in files:
        rel = path.relative_to(SRC).as_posix()
        if "/_benchmarks/" in "/" + rel or rel in ALLOWED:
            continue
        offenders += [f"{rel}:{line}: {text}" for line, text in _direct_calls(path)]
    assert not offenders, "direct public-shap computation; route it through mlframe.inference.tree_shap.TreeExplainer:\n  " + "\n  ".join(offenders)


def test_every_allowed_file_still_exists_and_still_calls_the_public_library():
    """An entry that no longer holds a direct call is stale: remove it so the list only ever shrinks."""
    stale = []
    for rel in ALLOWED:
        path = SRC / rel
        if not path.exists():
            stale.append(f"{rel}: file is gone")
        elif not _direct_calls(path):
            stale.append(f"{rel}: no direct call left")
    assert not stale, "stale entries in ALLOWED:\n  " + "\n  ".join(stale)
