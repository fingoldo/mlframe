"""Regression sensor for S47: ``_filter_polars_cat_features_by_dtype`` must be hoisted out of
the per-weight loop in ``_phase_train_one_target_body``.

The function result depends only on (prepared_train schema, _cat_features), both invariant across
weights. Re-calling it inside the weight loop pays a per-col dtype check on every iteration.

The hoist landed in an earlier wave. This sensor pins the location so a refactor that re-inlines
the call back into the weight loop (e.g. moving it into ``current_model_params`` build) trips a
clear failure rather than a silent perf regression.
"""

from __future__ import annotations

from pathlib import Path


def _read_phase_body() -> str:
    """Read the module holding the per-model / per-weight stage helpers, where the weight loop lives."""
    p = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "training" / "core" / "_phase_train_one_target_steps.py"
    return p.read_text(encoding="utf-8")


def test_S47_filter_polars_cat_features_by_dtype_hoisted_above_weight_loop():
    """The ``_filter_polars_cat_features_by_dtype`` call must run BEFORE the weight loop
    (``for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items()``). The call may sit in a stage helper
    carved out of the loop's function: then it is that helper's call SITE that has to precede the loop header.
    """
    import ast

    tree = ast.parse(_read_phase_body())
    weight_loops = [
        n.lineno for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and "weight_schemas.items()" in ast.unparse(n.iter)
    ]
    assert weight_loops, "could not locate weight_schemas loop in _phase_train_one_target_steps.py"
    loop_line = min(weight_loops)

    def _calls(name):
        """Lines where ``name(...)`` is called."""
        return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]

    def _enclosing(lineno):
        """Innermost function containing ``lineno``."""
        best = None
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= lineno <= n.end_lineno and (best is None or n.lineno > best.lineno):
                best = n
        return best

    filter_calls = _calls("_filter_polars_cat_features_by_dtype")
    assert filter_calls, "could not locate _filter_polars_cat_features_by_dtype call site"
    for fl in filter_calls:
        owner = _enclosing(fl)
        loop_owner = _enclosing(loop_line)
        if owner is not None and owner is not loop_owner:
            positions = _calls(owner.name)
            assert positions, f"{owner.name}, which holds the cat-feature filter, is never called"
        else:
            positions = [fl]
        for pos in positions:
            assert pos < loop_line, (
                f"_filter_polars_cat_features_by_dtype (via line {pos}) is reached AT or BELOW the weight loop header at line "
                f"{loop_line}; the filter must be hoisted above the loop to avoid per-weight invocations."
            )


def _assigns_subscript(tree, container: str, key: str):
    """Assign nodes of the form ``container[key] = ...``."""
    import ast

    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name)
            and t.value.id == container
            and isinstance(t.slice, ast.Constant)
            and t.slice.value == key
            for t in node.targets
        )
    ]


def test_S47_cb_extra_fit_invariant_carries_filter_result_into_loop():
    """The hoist must thread the filter result through ``_cb_extra_fit_invariant`` so the weight
    loop only stitches the precomputed dict into fit_params (no recompute)."""
    import ast

    tree = ast.parse(_read_phase_body())
    weight_loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and [e.id for e in n.target.elts if isinstance(e, ast.Name)] == ["weight_name", "weight_values"]
    ]
    assert len(weight_loops) == 1
    loop_nodes = {id(n) for n in ast.walk(weight_loops[0])}

    stores = _assigns_subscript(tree, "_cb_extra_fit_invariant", "cat_features")
    assert len(stores) == 1
    assert isinstance(stores[0].value, ast.Name) and stores[0].value.id == "_valid_cat_inv"
    assert id(stores[0]) not in loop_nodes, "the filter result must be stored before the weight loop"

    merges = _assigns_subscript(tree, "current_model_params", "fit_params")
    merged = [
        m
        for m in merges
        if isinstance(m.value, ast.Dict)
        and any(k is None and isinstance(v, ast.Name) and v.id == "_cb_extra_fit_invariant" for k, v in zip(m.value.keys, m.value.values))
    ]
    assert len(merged) == 1

    def _enclosing_function(node):
        """Innermost function containing ``node``."""
        best = None
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= node.lineno <= n.end_lineno and (best is None or n.lineno > best.lineno):
                best = n
        return best

    # the stitch happens per weight: either inside the loop itself or in a stage helper the loop body calls
    owner = _enclosing_function(merged[0])
    in_loop = id(merged[0]) in loop_nodes
    called_from_loop = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == owner.name for n in ast.walk(weight_loops[0]))
    stage_helpers_called_from_loop = {n.func.id for n in ast.walk(weight_loops[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    transitive = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == owner.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name in stage_helpers_called_from_loop
        for n in ast.walk(fn)
    )
    assert in_loop or called_from_loop or transitive, "the merge into fit_params must run once per weight iteration"


def test_S47_ngb_fallback_snapshot_cached_outside_loop():
    """Companion hoist for the NGBoost TypeError fallback: ``original_model.get_params(deep=False)``
    + dict-comprehension are invariant across weights; the snapshot must be cached once (on first
    use) so subsequent weight iterations splat the cached dict instead of re-paying ``get_params``.
    """
    import logging

    from mlframe.training.core._phase_train_one_target_schema import _clone_model_with_sticky_flags

    get_params_calls = []

    class _NgbLike:
        """``get_params`` exposes an attribute the constructor does not accept, so sklearn.clone raises TypeError."""

        def __init__(self, a=1, b=2):
            self.a = a
            self.b = b

        def get_params(self, deep=True):
            """Constructor params plus one extra, non-constructor entry."""
            get_params_calls.append(deep)
            return {"a": self.a, "b": self.b, "extra": 0}

    original = _NgbLike(a=7, b=9)
    reuse_calls = []

    def _forward(src, dst):
        """Record the cache hand-over between original and clone."""
        reuse_calls.append((src, dst))

    def _cached_init_params(cls):
        """Constructor parameter names of ``cls``."""
        return {"a", "b"}

    first, snapshot = _clone_model_with_sticky_flags(original, _cached_init_params, None, _forward, logging.getLogger(__name__))
    assert snapshot == {"a": 7, "b": 9}
    assert (first.a, first.b) == (7, 9) and first is not original
    after_first = len(get_params_calls)
    second, snapshot_again = _clone_model_with_sticky_flags(original, _cached_init_params, snapshot, _forward, logging.getLogger(__name__))
    assert snapshot_again is snapshot
    assert (second.a, second.b) == (7, 9) and second is not first
    # sklearn.clone reads get_params(deep=False) once per clone; the first call adds the snapshot read, the second must not.
    assert after_first == 2
    assert len(get_params_calls) - after_first == 1, "the snapshot must be built once and reused on the following weight iterations"
    assert reuse_calls == [(original, first), (original, second)]
