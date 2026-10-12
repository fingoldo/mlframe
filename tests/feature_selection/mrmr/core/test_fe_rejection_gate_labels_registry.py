"""Every FE rejection gate label written anywhere in the filters package must be registered, and every registered label must be written.

The ledger's ``gate`` column is only trustworthy if each drop site names the gate that actually killed the candidate.
A silent default such as ``rec.get("gate", "engineered_mi_prevalence")`` once attributed a different gate's drops to the
prevalence gate, so this test also rejects any ``.get("gate", ...)`` with a default.
"""

import ast
from pathlib import Path

import mlframe.feature_selection.filters as _filters_pkg
from mlframe.feature_selection.filters._fe_rejection_ledger import FE_GATE_LABELS

_FILTERS_ROOT = Path(_filters_pkg.__file__).parent
_SYNTHETIC_LABELS = {"ledger_capped"}


def _string_constants(expr):
    """Literal strings a gate expression can evaluate to: a constant, or both branches of a conditional."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {expr.value}
    if isinstance(expr, ast.IfExp):
        return _string_constants(expr.body) | _string_constants(expr.orelse)
    return set()


def _collect_gate_labels():
    """Return (labels written as literals, locations of silent ``.get("gate", default)`` fallbacks)."""
    labels: set[str] = set()
    silent_defaults: list[str] = []
    for path in _FILTERS_ROOT.rglob("*.py"):
        if "_benchmarks" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "gate":
                labels.update(_string_constants(node.value))
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == "gate":
                        labels.update(_string_constants(value))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "gate"
                and len(node.args) > 1
            ):
                silent_defaults.append(f"{path.relative_to(_FILTERS_ROOT)}:{node.lineno}")
    return labels, silent_defaults


def test_every_written_gate_label_is_registered():
    """A gate label written at a drop site must be one the ledger registry declares."""
    labels, _ = _collect_gate_labels()
    unregistered = labels - set(FE_GATE_LABELS)
    assert not unregistered, f"gate labels written but not in FE_GATE_LABELS: {sorted(unregistered)}"


def test_every_registered_gate_label_is_written():
    """A registered label no drop site writes is dead registry weight and hides a missing instrumentation point."""
    labels, _ = _collect_gate_labels()
    dead = set(FE_GATE_LABELS) - labels - _SYNTHETIC_LABELS
    assert not dead, f"registered gate labels never written by any drop site: {sorted(dead)}"


def test_no_silent_default_on_gate_label_lookup():
    """Reading a record's gate with a default would relabel a foreign drop as the default gate."""
    _, silent_defaults = _collect_gate_labels()
    assert not silent_defaults, f"silent gate-label default at: {silent_defaults}"
