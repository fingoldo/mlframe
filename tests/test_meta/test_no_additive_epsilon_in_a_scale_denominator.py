"""No production scale denominator may be guarded by adding a small constant to it.

``x / (std + 1e-12)`` reads as a guard against dividing by zero, but the constant is absolute while the quantity it pads is in the column's
own units. Above 1e-12 it does nothing; at or below it, it becomes the divisor and silently rescales the result instead of reporting that
there was nothing to divide by. The class has been fixed four times over (the basis axes, the FE normalisers, the stability-cluster
correlation matrix, the MI uplift ratio); this gate is what stops a fifth copy from landing.

The fix is always the same shape: compare the scale against the data's own magnitude (``_safe_scale.guarded_scale`` /
``scale_is_usable``) and treat "no usable scale" as its own outcome, rather than dividing by a pad.
"""

from __future__ import annotations

from tests.test_meta._scan_guard import assert_scanned_enough

import ast
import pathlib
import re

# Names whose value is a scale, i.e. a divisor carrying the data's units. A pad on any of these is the bug.
_SCALE_NAME = re.compile(r"(?i)(^|[^a-z])(std|stdev|sd|var|variance|norm|ptp|scale|span|spread|sigma|denom|baseline|range|width)([^a-z]|$)")
# Scoped to the selection surface: the families this class was fixed across, where a padded divisor changes what gets selected.
# Outside it the same shape occurs with a different meaning (a Kalman innovation variance, an optimizer step norm, a deliberate
# feature-definition ratio), which this gate is not equipped to judge.
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection"
# Benchmarks and synthetic-data generators divide fixture columns they construct themselves at unit scale, where the pad is inert and no
# selection depends on it. Everything that scores, ranks or gates is in scope.
_OUT_OF_SCOPE = ("_benchmarks",)


def _offending_divisions(tree: ast.AST):
    """Every ``_ / (<scale name> + <small float>)`` in the module, as (line, source) pairs."""
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        right = node.right
        if not (isinstance(right, ast.BinOp) and isinstance(right.op, ast.Add)):
            continue
        pad = right.right
        if not (isinstance(pad, ast.Constant) and isinstance(pad.value, float) and 0.0 < pad.value < 1e-3):
            continue
        if _SCALE_NAME.search(ast.unparse(right.left)):
            found.append((node.lineno, ast.unparse(node)[:160]))
    return found


def test_no_additive_epsilon_in_a_scale_denominator():
    """AST-walk the package and fail on any padded scale divisor, naming every site."""
    offences = []
    _files = sorted(_SRC.rglob("*.py"))
    assert_scanned_enough(len(_files), str(_SRC), minimum=50)
    for path in _files:
        if any(part in _OUT_OF_SCOPE for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # a file this interpreter cannot parse is not this gate's business
            continue
        for lineno, src in _offending_divisions(tree):
            offences.append(f"{path.relative_to(_SRC)}:{lineno}: {src}")
    assert not offences, "padded scale denominators (divide by the guarded scale instead, see _safe_scale):\n" + "\n".join(offences)


def test_the_gate_detects_the_pattern_it_is_meant_to_catch():
    """The detector must actually fire: the shape it targets, and not the shapes it must leave alone."""
    caught = _offending_divisions(ast.parse("z = (x - mean) / (std + 1e-12)"))
    assert len(caught) == 1, "the gate no longer detects the padded z-score it exists for"
    for benign in (
        "z = (x - mean) / std",  # the fix
        "z = (x - lo) / (hi - lo)",  # an unpadded span
        "r = numerator / (count + 1)",  # a count, not a scale, and not a small pad
        "w = a / (weight_sum + 1e-12)",  # a sum of weights is not a scale in the data's units
    ):
        assert not _offending_divisions(ast.parse(benign)), f"the gate fires on a benign form: {benign}"
