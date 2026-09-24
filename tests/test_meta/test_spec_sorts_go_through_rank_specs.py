"""Every keyed sort in composite discovery and the composite phases goes through ``rank_specs`` or is listed with why it
does not order specs.

``rank_specs`` refuses to order scores of different units, splits or estimators, or a score measured for another transform:
the defect class behind the cross-target budget sorting RMSE fractions against MI nats and the rerank mixing honest and CV
RMSE. A bare ``sorted(..., key=...)``, ``.sort(key=...)`` or ``np.lexsort`` bypasses that check, so each one is listed here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mlframe

_TRAINING = Path(mlframe.__file__).resolve().parent / "training"

# (file, enclosing function) -> why the sort does not rank specs.
ALLOWED = {
    ("__init__.py", "_rank_bases_by_mi_for_cap"): "base columns by MI(y, base), before any spec exists",
    ("_auto_base.py", "_auto_base"): "base columns by their auto-base score, and a log preview of demotions",
    ("_filter.py", "_filter_features"): "leak-dropped feature columns by |corr| for the log line",
    ("_grouped_causal_bases.py", "_segment_order"): "row order by (group, time) for causal bases",
    ("_interaction_bases.py", "score_interaction_pairs"): "feature-pair candidates by MI gain, before they become bases",
    ("_per_base_x.py", "base_ordered"): "spec indices grouped by base for cache locality; results return to spec order",
    ("auto_detect.py", "detect_time_column_candidates"): "candidate time columns by detection score",
    ("auto_detect.py", "detect_group_column_candidates"): "candidate group columns by detection score",
    ("auto_detect.py", "detect_cat_columns"): "candidate categorical columns by detection score",
    ("_phase_composite_discovery_gates.py", "_per_target_discovery_config"): "feature-ablation hints by delta_pct",
}


def keyed_sorts(source: str) -> set[str]:
    """Enclosing function names of the keyed sorts in ``source``."""
    found: set[str] = set()
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            f, kw = node.func, {k.arg for k in node.keywords}
            if ((isinstance(f, ast.Name) and f.id == "sorted" and "key" in kw) or (isinstance(f, ast.Attribute) and f.attr == "sort" and "key" in kw)
                    or (isinstance(f, ast.Attribute) and f.attr == "lexsort")):
                found.add(stack[-1] if stack else "<module>")
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return found


def _scanned_files() -> list[Path]:
    discovery = [p for p in (_TRAINING / "composite" / "discovery").glob("*.py") if p.name != "_score.py"]
    return sorted(discovery + list((_TRAINING / "core").glob("_phase_composite*.py")))


def test_every_keyed_sort_is_rank_specs_or_listed():
    hits = {(p.name, fn) for p in _scanned_files() for fn in keyed_sorts(p.read_text(encoding="utf-8"))}
    unlisted = sorted(hits - set(ALLOWED))
    assert not unlisted, f"route these through discovery._score.rank_specs, or list them with the reason they do not rank specs: {unlisted}"
    stale = sorted(set(ALLOWED) - hits)
    assert not stale, f"no keyed sort left here; drop the entry: {stale}"


def test_the_scan_sees_each_sort_shape():
    src = '''
def a(specs):
    return sorted(specs, key=lambda s: s.mi_gain)
def b(specs):
    specs.sort(key=lambda s: s.rmse)
def c(names, scores):
    return np.lexsort((names, scores))
def d(names):
    return sorted(names)
'''
    assert keyed_sorts(src) == {"a", "b", "c"}
