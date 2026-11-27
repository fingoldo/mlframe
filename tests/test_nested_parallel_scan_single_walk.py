"""The single-walk ``_function_facts`` gives what the per-question helpers it replaced give, on every shape the scanner reads.

The package build walked each function's subtree about ten times, once per question (what does it call, is the call guarded, does it start a
pool, does it touch a dispatch table). ``_function_facts`` answers all of them from one list of nodes. These cases pin that the answers did not move.
"""

from __future__ import annotations

import ast

import pytest

from mlframe import _nested_parallel_scan as scan

_CASES = {
    "plain_calls": "def f():\n    a()\n    b.c()\n",
    "guarded_call_is_not_an_edge": "def f():\n    with parallel_kernel_entry():\n        kernel()\n    other()\n",
    "attribute_guard": "def f():\n    with guard.parallel_kernel_entry():\n        kernel()\n",
    "thread_pool_executor": "def f():\n    ThreadPoolExecutor(2).submit(work)\n",
    "joblib_parallel": "def f():\n    Parallel(n_jobs=2)(delayed(work)(i) for i in range(3))\n",
    "delayed_name_is_a_callee": "def f():\n    run(delayed(work), delayed(other))\n",
    "parallel_run_shim": "def f():\n    parallel_run(work)\n",
    "dispatch_table_reference": "TABLE = {'a': k1, 'b': k2}\n\ndef f(name):\n    fn = TABLE[name]\n    return fn()\n",
    "dispatch_table_with_guard": "TABLE = {'a': k1}\n\ndef f(name):\n    with parallel_kernel_entry():\n        TABLE[name]()\n",
    "nested_function_calls_count": "def f():\n    def inner():\n        deep()\n    inner()\n",
    "async_function": "async def f():\n    with parallel_kernel_entry():\n        await k()\n    other()\n",
}


def _functions(source: str):
    """The module's dispatch tables and every function definition in ``source``."""
    tree = ast.parse(source)
    tables = scan._dispatch_tables(tree)
    return tables, [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _reference_starts_thread_pool(node: ast.AST) -> bool:
    """The per-question check the single walk replaced: a ``ThreadPoolExecutor``, or any joblib ``Parallel`` / ``parallel_run`` call."""
    if "ThreadPoolExecutor" in scan._called_names(node):
        return True
    return any(isinstance(call, ast.Call) and (getattr(call.func, "id", None) or getattr(call.func, "attr", None)) in scan._FAN_OUT_CALLS for call in ast.walk(node))


def _reference_edges(node: ast.AST, tables) -> set:
    """What the package build computed before the single walk: ``_called_names`` plus the dispatch-table edges."""
    edges = scan._called_names(node)
    if not scan._guarded_call_nodes(node):
        referenced = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        for table, members in tables.items():
            if table in referenced:
                edges |= members
    return edges


@pytest.mark.parametrize("case", sorted(_CASES))
def test_single_walk_edges_and_pool_flag_match_the_per_question_helpers(case):
    """Each case gives the same edge set and pool flag from the single walk as from the helpers it replaced."""
    tables, functions = _functions(_CASES[case])
    assert functions, "the case must define a function"
    for node in functions:
        edges, starts_pool = scan._function_facts(node, tables)
        assert edges == _reference_edges(node, tables), f"{case}:{node.name} edges moved"
        assert starts_pool == _reference_starts_thread_pool(node), f"{case}:{node.name} pool flag moved"


def test_a_guarded_dispatch_table_adds_no_edges_and_an_unguarded_one_adds_its_members():
    """The over-approximation is kept: a function that guards anything is taken to guard its dispatch too."""
    tables, (unguarded,) = _functions(_CASES["dispatch_table_reference"])
    edges, _ = scan._function_facts(unguarded, tables)
    assert {"k1", "k2"} <= edges
    tables, (guarded,) = _functions(_CASES["dispatch_table_with_guard"])
    edges, _ = scan._function_facts(guarded, tables)
    assert "k1" not in edges
