"""Meta-test: a test module must not construct ``shap.TreeExplainer(...)`` outside a
``with _maybe_patch_shap_xgb_base_score():`` block.

On shap<0.52, ``_maybe_patch_shap_xgb_base_score`` scopes a monkeypatch of
``shap.explainers._tree.float`` around the ``TreeExplainer`` call that needs it (XGBoost 2.x/3.x
base_score parsing) and restores it on exit. A raw, unwrapped ``TreeExplainer(...)`` call leaves
NO protection: if an unpatched XGBoost TreeExplainer runs first in a pytest-xdist worker, it can
also be corrupted by a PRIOR test's stale patch, or leave one behind for the NEXT test in the same
worker (observed live: ``AttributeError: 'TreeEnsemble' object has no attribute 'values'`` on a
LightGBM TreeExplainer that ran after an earlier unwrapped call -- see
test_shap_xgb_patch_version_gate.py's docstring for the full incident, and
f526b6b71/the follow-up commit fixing the same file's own ``test_shap_proxy_treeshap_interactions.py``
a SECOND time after a first sweep missed a call written as ``__import__("shap").TreeExplainer(...)``
instead of the more common ``import shap; shap.TreeExplainer(...)`` -- a plain text/grep sweep is not
reliable against arbitrary call-site styles, which is why this meta-test exists: it matches on the
AST attribute name ``TreeExplainer`` regardless of how the ``shap`` module reference was obtained).

Baseline-diff (not zero-tolerance) so a deliberately-unwrapped call for a rare, confirmed-safe
scenario (e.g. shap>=0.52 only, where the patch is already a no-op) can be reviewed and
grandfathered rather than blocking unrelated work. Refresh with
``--refresh-unprotected-treeexplainer-baseline`` after review.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_unprotected_treeexplainer_baseline.json"

_GUARD_NAME = "_maybe_patch_shap_xgb_base_score"


def _is_guard_with(node: ast.With) -> bool:
    """True if this ``with`` statement's context expression calls the guard (by name, regardless
    of how it was imported/aliased on the receiver -- ``spe._maybe_patch_shap_xgb_base_score()``,
    ``_maybe_patch_shap_xgb_base_score()``, etc., all match on the attribute/name itself)."""
    for item in node.items:
        expr = item.context_expr
        if not isinstance(expr, ast.Call):
            continue
        f = expr.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        if name == _GUARD_NAME:
            return True
    return False


def _is_tree_explainer_call(node: ast.AST) -> bool:
    """True for ``<anything>.TreeExplainer(...)`` -- matches regardless of how ``shap`` was
    referenced (``shap.TreeExplainer``, ``__import__("shap").TreeExplainer``, an aliased import,
    ...)."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "TreeExplainer"


def _unprotected_treeexplainer_lines(tree: ast.Module) -> list[int]:
    """Line numbers of every ``TreeExplainer(...)`` call not nested inside a guard ``with`` block."""
    out: list[int] = []

    def walk(node: ast.AST, protected: bool) -> None:
        """Depth-first walk, tracking whether the current node is inside a guard ``with`` block."""
        if isinstance(node, ast.With) and _is_guard_with(node):
            protected = True
        if _is_tree_explainer_call(node) and not protected:
            out.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            walk(child, protected)

    walk(tree, protected=False)
    return out


def _build_offending_set() -> set[str]:
    """``{"relpath:lineno", ...}`` for every unprotected TreeExplainer(...) call under ``tests/``."""
    out: set[str] = set()
    for py in _TESTS_DIR.rglob("test_*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = py.relative_to(_TESTS_DIR).as_posix()
        for lineno in _unprotected_treeexplainer_lines(tree):
            out.add(f"{rel}:{lineno}")
    return out


def _refresh_requested() -> bool:
    """True if ``--refresh-unprotected-treeexplainer-baseline`` was passed on the pytest command line."""
    return "--refresh-unprotected-treeexplainer-baseline" in sys.argv


def test_no_new_unprotected_shap_treeexplainer():
    """No test module gains a new TreeExplainer(...) call outside the guard with-block."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"unprotected-TreeExplainer baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    drained = sorted(baseline - current)

    if drained:
        sys.stderr.write(
            f"\n[test_no_new_unprotected_shap_treeexplainer] {len(drained)} site(s) DRAINED:\n  "
            + "\n  ".join(drained)
            + "\n  Refresh: pytest ... --refresh-unprotected-treeexplainer-baseline\n"
        )

    assert not added, (
        f"{len(added)} new unprotected shap.TreeExplainer(...) construction(s). Wrap in "
        "`with _maybe_patch_shap_xgb_base_score():` (import from "
        "mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain) -- otherwise an unpatched "
        "shap<0.52 XGBoost TreeExplainer can leak a monkeypatch into (or be corrupted by) whatever "
        "TreeExplainer runs next in the same pytest-xdist worker, regardless of model type. If this is a "
        "genuine, reviewed exception, re-run with --refresh-unprotected-treeexplainer-baseline.\n  " + "\n  ".join(added)
    )


_DETECTOR_SAMPLE = '''
import shap
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain as spe

def unprotected():
    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

def unprotected_import_call_style():
    phi = __import__("shap").TreeExplainer(model).shap_values(X)

def protected():
    with spe._maybe_patch_shap_xgb_base_score():
        ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

def protected_bare_name():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _maybe_patch_shap_xgb_base_score
    with _maybe_patch_shap_xgb_base_score():
        ex = shap.TreeExplainer(model)
'''


def test_detector_flags_both_unprotected_styles_and_ignores_protected_ones():
    """The scan flags both the attribute-call and __import__ styles when unprotected, and ignores
    both protected variants (aliased-module and bare-name guard call)."""
    found = _unprotected_treeexplainer_lines(ast.parse(_DETECTOR_SAMPLE))
    assert found == [6, 9], found
