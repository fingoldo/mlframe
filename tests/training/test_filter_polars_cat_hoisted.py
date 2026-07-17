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


def _read_phase_body_with_siblings() -> str:
    """Concat ``_phase_train_one_target_body.py`` + sibling files for source-grep
    checks that need code carved out into themed siblings (e.g. the NGBoost
    fallback snapshot lazy-init now lives in ``_phase_train_one_target_schema.py``).
    Use the bare-body reader for sensors that depend on line-number ordering.
    """
    _core = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "training" / "core"
    src = (_core / "_phase_train_one_target_body.py").read_text(encoding="utf-8")
    for sib_name in (
        "_phase_train_one_target_schema.py",
        "_phase_train_one_target_helpers.py",
        "_phase_train_one_target_model_setup.py",
        "_phase_train_one_target_ensembling.py",
    ):
        sib = _core / sib_name
        if sib.exists():
            src += "\n" + sib.read_text(encoding="utf-8")
    return src


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


def test_S47_cb_extra_fit_invariant_carries_filter_result_into_loop():
    """The hoist must thread the filter result through ``_cb_extra_fit_invariant`` so the weight
    loop only stitches the precomputed dict into fit_params (no recompute)."""
    src = _read_phase_body()
    assert '_cb_extra_fit_invariant["cat_features"] = _valid_cat_inv' in src, (
        "the hoist contract is: filter result lands in _cb_extra_fit_invariant['cat_features']; "
        "the weight loop then merges _cb_extra_fit_invariant into current_model_params['fit_params']."
    )
    assert 'current_model_params["fit_params"] = {**current_model_params["fit_params"], **_cb_extra_fit_invariant}' in src, (
        "the weight loop must merge _cb_extra_fit_invariant into fit_params instead of re-running the filter."
    )


def test_S47_ngb_fallback_snapshot_cached_outside_loop():
    """Companion hoist for the NGBoost TypeError fallback: ``original_model.get_params(deep=False)``
    + dict-comprehension are invariant across weights; the snapshot must be cached once (on first
    use) so subsequent weight iterations splat the cached dict instead of re-paying ``get_params``.
    """
    src = _read_phase_body_with_siblings()
    assert "_ngb_fallback_snapshot: dict | None = None" in src, "expected lazy-init _ngb_fallback_snapshot pinned outside the weight loop"
    assert "if _ngb_fallback_snapshot is None:" in src, "weight loop must lazily populate the snapshot on first TypeError hit; subsequent iters reuse it"
