"""Meta-test: ctx-derived kwargs MUST be forwarded to inner-function call sites.

Complements `test_setup_configuration_propagation.py` (which guards the
**constructor layer**: public-API kwarg -> ctx slot). This guards the
**call-site layer**: ctx slot -> inner-function kwarg at the orchestrator call.

The bug class this catches: an inner function (e.g. ``score_ensemble``,
``compute_oof_holdout_predictions``, ``RecurrentClassifierWrapper.fit``,
``compute_dummy_baselines``) ACCEPTS a kwarg like ``group_ids`` / ``sample_weight`` /
``time_ordering`` -- but the orchestrator that has the corresponding ctx slot
forgets to pass it. The inner function silently uses its default (None ->
"no group awareness", "i.i.d. rows", "random shuffle"). Behavioural tests don't
catch this: the model still trains, metrics still look plausible. Per-function
unit tests don't catch this either: they call the inner fn with explicit kwargs.
The gap is at the boundary.

The check parses the orchestrator modules and inspects the syntax tree: a contract names the
callee, the keyword it must receive and the expression that feeds it, so renaming a comment or
reflowing a call does not matter but dropping or re-pointing the keyword does.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import mlframe as _mlframe

_CORE = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core"


def _callee_name(func: ast.expr) -> str:
    """Bare name of a call target, ``f`` for ``f(...)`` and ``m.f(...)``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _core_trees() -> list[ast.Module]:
    """Parsed syntax trees of every module under ``training/core``."""
    return [ast.parse(p.read_text(encoding="utf-8")) for p in sorted(_CORE.rglob("*.py"))]


def _has_call_keyword(callee: str, kw: str, value: str) -> bool:
    """Whether some ``callee(..., kw=<value>)`` call exists under ``training/core``."""
    for tree in _core_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee_name(node.func) == callee:
                for k in node.keywords:
                    if k.arg == kw and ast.unparse(k.value) == value:
                        return True
    return False


def _has_dict_entry(kw: str, value: str, module: str) -> bool:
    """Whether the named module builds a dict literal entry ``{kw: <value>}``."""
    tree = ast.parse((_CORE / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == kw and ast.unparse(v) == value:
                    return True
    return False


def _has_subscript_assign(target: str, key: str, value: str, module: str) -> bool:
    """Whether the named module assigns ``target[key] = <value>``."""
    tree = ast.parse((_CORE / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and ast.unparse(node.value) == value:
            for t in node.targets:
                if isinstance(t, ast.Subscript) and ast.unparse(t.value) == target and isinstance(t.slice, ast.Constant) and t.slice.value == key:
                    return True
    return False


def _function_param_default(module: str, func: str, param: str) -> tuple[bool, str | None]:
    """``(param exists, unparsed default or None)`` for ``func`` in the named module."""
    tree = ast.parse((_CORE / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            args = node.args
            pos = args.posonlyargs + args.args
            defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
            for a, d in list(zip(pos, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults)):
                if a.arg == param:
                    return True, (ast.unparse(d) if d is not None else None)
    return False, None


_GROUP_IDS_FROM_CTX = "getattr(ctx, 'group_ids', None)"

_CONTRACTS = [
    ("call", "score_ensemble", "group_ids", _GROUP_IDS_FROM_CTX, "simple-path score_ensemble missing group_ids from ctx"),
    ("call", "score_ensemble", "sample_weight", "_ens_sample_weight", "simple-path score_ensemble missing sample_weight from ctx"),
    ("param", "_phase_dummy_baselines.py", "run_dummy_baselines", "group_ids", "run_dummy_baselines must accept group_ids in its signature"),
    ("call", "compute_dummy_baselines", "group_ids_train", "_gid_train", "compute_dummy_baselines call missing group_ids_train"),
    ("call", "run_dummy_baselines", "group_ids", _GROUP_IDS_FROM_CTX, "run_dummy_baselines caller missing group_ids=ctx.group_ids"),
    ("dict", "_phase_recurrent.py", "group_ids", _GROUP_IDS_FROM_CTX, "recurrent-rerun score_ensemble missing group_ids"),
    ("dict", "_phase_recurrent.py", "sample_weight", "_sw_for_target", "recurrent-rerun score_ensemble missing sample_weight"),
    ("assign", "_phase_recurrent.py", "_fit_kwargs", "sample_weight", "recurrent .fit missing sample_weight propagation"),
    ("call", "compute_oof_holdout_predictions", "time_ordering", "_time_ordering", "OOF holdout call missing time_ordering=ctx.timestamps[idx]"),
    ("call", "compute_oof_holdout_predictions", "sample_weight", "_sw_for_oof", "OOF holdout call missing sample_weight=ctx.sample_weights"),
]


def _holds(contract: tuple) -> bool:
    """Evaluate one contract against the parsed orchestrator modules."""
    kind = contract[0]
    if kind == "call":
        _, callee, kw, value, _why = contract
        return _has_call_keyword(callee, kw, value)
    if kind == "dict":
        _, module, kw, value, _why = contract
        return _has_dict_entry(kw, value, module)
    if kind == "assign":
        _, module, target, key, _why = contract
        return _has_subscript_assign(target, key, "_sw_for_target", module)
    _, module, func, param, _why = contract
    found, default = _function_param_default(module, func, param)
    return found and default == "None"


@pytest.mark.parametrize("contract", _CONTRACTS, ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}-{c[3]}")
def test_ctx_kwarg_propagates_to_inner_call_site(contract):
    """The orchestrator call site passes the ctx-derived keyword to its callee, verified on the syntax tree."""
    assert _holds(contract), f"propagation regression: {contract[-1]}"


def test_contract_checker_rejects_a_dropped_keyword():
    """The checker is not vacuous: a call without the keyword, or with a different value, does not satisfy a contract."""
    assert not _has_call_keyword("score_ensemble", "group_ids", "definitely_not_the_ctx_slot")
    assert not _has_call_keyword("no_such_callee_anywhere", "group_ids", _GROUP_IDS_FROM_CTX)
    assert not _has_dict_entry("group_ids", _GROUP_IDS_FROM_CTX, "_phase_dummy_baselines.py")


def test_meta_test_self_check():
    """The contract list is not empty, otherwise the parametrize gives false confidence."""
    assert len(_CONTRACTS) >= 10, f"contract table looks short ({len(_CONTRACTS)} entries)"
