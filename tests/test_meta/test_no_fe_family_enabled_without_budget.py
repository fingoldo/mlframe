"""Meta-test: a test must not switch an FE family on while pinning ``fe_max_steps=0``.

``fe_max_steps=0`` is the unconditional "no feature engineering at all" contract - it outranks every
``fe_*_enable`` flag. Combining the two in one construction asks for a family and then switches it off, and
the result is silent: the fit succeeds, the roster of engineered columns is simply empty, and the assertion
about that roster fails for a reason that looks nothing like its cause. Where the assertion is a threshold
rather than a membership check, it can even keep passing while measuring a raw-only fit.

Three separate test factories had drifted into this, each using ``fe_max_steps=0`` in an older sense - "no
pair step", "isolate the general-FE competitors", "keep the run cheap" - none of which the knob means now.
The clearest symptom was four different mechanisms in one file reporting a bit-identical +0.0048 lift,
because all four were measuring the same raw-only fit.

Both spellings are caught: a direct ``MRMR(...)`` call, and a ``dict(...)``/dict-literal preset that is
later splatted into one. The check is deliberately syntactic - it looks at literal keyword arguments in a
single call - so a factory that computes the budget at runtime (the correct fix: grant a budget when the
caller explicitly opts a family in) is not flagged.

Refresh with::

    pytest tests/test_meta/test_no_fe_family_enabled_without_budget.py --refresh-fe-budget-conflict-baseline
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_fe_budget_conflict_baseline.json"


def _refresh_requested() -> bool:
    """True if ``--refresh-fe-budget-conflict-baseline`` was passed on the pytest command line."""
    return "--refresh-fe-budget-conflict-baseline" in sys.argv


def _is_zero(node: ast.AST) -> bool:
    """True for a literal ``0`` (the only value that trips the no-FE contract)."""
    return isinstance(node, ast.Constant) and node.value == 0 and node.value is not False


def _is_truthy_literal(node: ast.AST) -> bool:
    """True for a literal that switches a flag ON. A non-literal is left alone - it may be a test parameter."""
    return isinstance(node, ast.Constant) and node.value is True


def _conflicts_in_call(node: ast.Call) -> list[str]:
    """``fe_*_enable`` flags set to a literal True in the same call that pins ``fe_max_steps=0``."""
    budget_zero = False
    enabled: list[str] = []
    for kw in node.keywords:
        if kw.arg is None:
            continue
        if kw.arg == "fe_max_steps" and _is_zero(kw.value):
            budget_zero = True
        elif kw.arg.startswith("fe_") and kw.arg.endswith("_enable") and _is_truthy_literal(kw.value):
            enabled.append(kw.arg)
    return enabled if budget_zero else []


def _conflicts_in_dict(node: ast.Dict) -> list[str]:
    """Same check for a dict LITERAL preset that is later splatted into a constructor."""
    budget_zero = False
    enabled: list[str] = []
    for k, v in zip(node.keys, node.values):
        if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
            continue
        if k.value == "fe_max_steps" and _is_zero(v):
            budget_zero = True
        elif k.value.startswith("fe_") and k.value.endswith("_enable") and _is_truthy_literal(v):
            enabled.append(k.value)
    return enabled if budget_zero else []


def _build_offending_set() -> set[str]:
    """``{"relpath:lineno:flag", ...}`` for every construction that enables a family under a zero budget."""
    out: set[str] = set()
    for py in _TESTS_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = py.relative_to(_TESTS_DIR).as_posix()
        for node in ast.walk(tree):
            flags: list[str] = []
            if isinstance(node, ast.Call):
                flags = _conflicts_in_call(node)
            elif isinstance(node, ast.Dict):
                flags = _conflicts_in_dict(node)
            for flag in flags:
                out.add(f"{rel}:{node.lineno}:{flag}")
    return out


def test_no_new_fe_family_enabled_under_a_zero_budget():
    """No construction switches an FE family on while pinning ``fe_max_steps=0``, beyond the baseline."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"fe-budget-conflict baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    drained = sorted(baseline - current)

    if drained:
        sys.stderr.write(
            f"\n[test_no_new_fe_family_enabled_under_a_zero_budget] {len(drained)} site(s) DRAINED:\n  "
            + "\n  ".join(drained)
            + "\n  Refresh: pytest ... --refresh-fe-budget-conflict-baseline\n"
        )

    assert not added, (
        f"{len(added)} construction(s) switch an FE family on while pinning fe_max_steps=0, which is the "
        "unconditional no-FE contract and outranks the flag. The family never runs, so the roster of "
        "engineered columns is empty and the assertion about it fails for a reason unrelated to its message "
        "- or, if the assertion is a threshold, quietly keeps passing while measuring a raw-only fit. Give "
        "the family the budget it needs and express whatever fe_max_steps=0 was standing in for (no pair "
        "step, isolating competitors) with the flags that name it.\n  " + "\n  ".join(added)
    )


_SAMPLE_CONFLICT = """
MRMR(fe_max_steps=0, fe_hybrid_orth_enable=True)
MRMR(fe_max_steps=1, fe_hybrid_orth_enable=True)
MRMR(fe_max_steps=0, fe_hybrid_orth_enable=False)
kw = dict(fe_max_steps=0, fe_rankgauss_enable=True)
kw2 = {"fe_max_steps": 0, "fe_spline_enable": True}
MRMR(fe_max_steps=0, fe_hybrid_orth_enable=flag)
"""


def test_detector_flags_only_the_real_conflicts():
    """Only the zero-budget-plus-enabled combinations are reported; a real budget or a non-literal is not.

    Without this the gate would look healthy forever if the matcher broke, since a baseline-diff test is
    green exactly when it finds nothing new.
    """
    tree = ast.parse(_SAMPLE_CONFLICT)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            found |= {(node.lineno, f) for f in _conflicts_in_call(node)}
        elif isinstance(node, ast.Dict):
            found |= {(node.lineno, f) for f in _conflicts_in_dict(node)}
    assert found == {
        (2, "fe_hybrid_orth_enable"),
        (5, "fe_rankgauss_enable"),
        (6, "fe_spline_enable"),
    }, found
