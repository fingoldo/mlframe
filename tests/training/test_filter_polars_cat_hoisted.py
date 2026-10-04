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
    """Read phase body."""
    p = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "training" / "core" / "_phase_train_one_target_body.py"
    return p.read_text(encoding="utf-8")


def test_S47_filter_polars_cat_features_by_dtype_hoisted_above_weight_loop():
    """The ``_filter_polars_cat_features_by_dtype`` call must appear BEFORE the weight loop
    (``for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items()``). Behavioural
    proxy: line numbers measured against the same source file.
    """
    src = _read_phase_body()
    lines = src.splitlines()
    # Find the indices.
    weight_loop_lines = [i for i, line in enumerate(lines) if "for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items()" in line]
    filter_call_lines = [i for i, line in enumerate(lines) if "_filter_polars_cat_features_by_dtype(prepared_train" in line]
    assert weight_loop_lines, "could not locate weight_schemas loop in _phase_train_one_target_body.py"
    assert filter_call_lines, "could not locate _filter_polars_cat_features_by_dtype call site"
    # Every filter call must be above the weight loop header.
    for fl in filter_call_lines:
        assert fl < min(weight_loop_lines), (
            f"_filter_polars_cat_features_by_dtype at line {fl + 1} appears AT or BELOW the weight loop "
            f"header at line {min(weight_loop_lines) + 1}; the filter must be hoisted above the loop "
            f"to avoid per-weight invocations."
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
    merged_in_loop = [
        m
        for m in merges
        if id(m) in loop_nodes
        and isinstance(m.value, ast.Dict)
        and any(k is None and isinstance(v, ast.Name) and v.id == "_cb_extra_fit_invariant" for k, v in zip(m.value.keys, m.value.values))
    ]
    assert len(merged_in_loop) == 1


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
