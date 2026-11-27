"""Cluster 41 regression: module-level duplicates the audit named are gone (audit 2026-11-20, 41-F42)."""

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "training"


def _assigned_names(path: Path) -> Counter:
    """Count module-level assignments per name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: Counter = Counter()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names[t.id] += 1
    return names


def test_logger_and_not_aggregatable_are_defined_once():
    """logger and _NOT_AGGREGATABLE are each assigned once per module (41-F42)."""
    targets = [(rel, name) for rel, wanted in (("reporting/_reporting_probabilistic_helpers.py", ("logger", "_NOT_AGGREGATABLE")), ("ranking/_ranker_suite_train_helpers.py", ("logger",))) for name in wanted]
    assert len(targets) == 3, "three assignments are checked: two in the probabilistic helpers, one in the ranker helpers"
    for rel, name in targets:
        assert _assigned_names(ROOT / rel)[name] == 1, (rel, name)


def test_slug_pattern_lives_in_one_module_and_dataclass_is_not_doubled():
    """The slug regex is defined only in strategies/__init__, and TemporalAuditResult has a single @dataclass (41-F42)."""
    assert _assigned_names(ROOT / "strategies/base.py")["_SLUG_PATTERN"] == 0
    assert _assigned_names(ROOT / "strategies/__init__.py")["_SLUG_PATTERN"] == 1
    tree = ast.parse((ROOT / "targets/_target_temporal_types.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TemporalAuditResult")
    assert len(cls.decorator_list) == 1
