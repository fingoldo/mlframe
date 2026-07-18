"""Sensor: _phase_train_one_target_body.py W14A partial carve preserves identity + helpers are wired correctly.

Carve lifts three pure helpers from ``_train_one_target`` into ``_phase_train_one_target_schema.py``:
- ``_build_and_record_model_schema`` (per-model schema record + n_classes / multilabel_strategy + FS report cache + metadata persist)
- ``_clone_model_with_sticky_flags`` (sklearn.clone + NGBoost / CatBoost fallbacks + _mlframe_posthoc_calibrate / _mlframe_polars_fastpath_broken re-assert + dataset-reuse cache forward)
- ``_resolve_weight_schemas_and_warn_val_placement`` (sample_weights -> weight_schemas dict + per-suite SW-LOG + VAL-PLACE-WARN banners)

The W12B-deferred deeper monolith carve (878 LOC -> <700) is NOT done in this wave; the body is still ~1069 LOC due to deeply-nested closure-captured locals that the helpers here don't reach.
"""

from __future__ import annotations

from types import SimpleNamespace


def test_w14a_phase_train_one_target_schema_identity():
    """W14a phase train one target schema identity."""
    from mlframe.training.core import _phase_train_one_target_body as body
    from mlframe.training.core import _phase_train_one_target_schema as schema

    assert body._build_and_record_model_schema is schema._build_and_record_model_schema
    assert body._clone_model_with_sticky_flags is schema._clone_model_with_sticky_flags
    assert body._resolve_weight_schemas_and_warn_val_placement is schema._resolve_weight_schemas_and_warn_val_placement


def test_w14a_phase_train_one_target_body_smoke_import():
    """The body module imports cleanly post-carve."""
    from mlframe.training.core._phase_train_one_target_body import _train_one_target

    assert callable(_train_one_target)


def test_w14a_resolve_weight_schemas_uniform_default(caplog):
    """No sample_weights -> uniform schema; banner emitted once per suite via ctx._sw_log_emitted latch."""
    import logging
    from mlframe.training.core._phase_train_one_target_schema import _resolve_weight_schemas_and_warn_val_placement

    ctx = SimpleNamespace(_sw_log_emitted=False, _val_placement_warn_emitted=False)
    split_config = SimpleNamespace(val_placement="forward")
    caplog.set_level(logging.INFO, logger="mlframe.training.core._phase_train_one_target")
    res = _resolve_weight_schemas_and_warn_val_placement(None, split_config, ctx)
    assert res == {"uniform": None}
    assert ctx._sw_log_emitted is True

    # Second call: latch is sticky, no additional banner
    caplog.clear()
    res2 = _resolve_weight_schemas_and_warn_val_placement(None, split_config, ctx)
    assert res2 == {"uniform": None}
    assert not any("default" in rec.message.lower() for rec in caplog.records)


def test_w14a_resolve_weight_schemas_backward_val_warns_on_non_uniform(caplog):
    """Backward val + non-uniform schema triggers the val_placement warning."""
    import logging
    from mlframe.training.core._phase_train_one_target_schema import _resolve_weight_schemas_and_warn_val_placement

    ctx = SimpleNamespace(_sw_log_emitted=False, _val_placement_warn_emitted=False)
    split_config = SimpleNamespace(val_placement="backward")
    caplog.set_level(logging.WARNING, logger="mlframe.training.core._phase_train_one_target")
    res = _resolve_weight_schemas_and_warn_val_placement(
        {"recency": [1.0, 2.0, 3.0]},
        split_config,
        ctx,
    )
    assert res == {"recency": [1.0, 2.0, 3.0]}
    assert ctx._val_placement_warn_emitted is True
    assert any("val_placement='backward'" in rec.message for rec in caplog.records)


def test_w14a_resolve_weight_schemas_forward_val_no_warn(caplog):
    """Forward val placement (default) does NOT trigger the warning even with non-uniform schemas."""
    import logging
    from mlframe.training.core._phase_train_one_target_schema import _resolve_weight_schemas_and_warn_val_placement

    ctx = SimpleNamespace(_sw_log_emitted=False, _val_placement_warn_emitted=False)
    split_config = SimpleNamespace(val_placement="forward")
    caplog.set_level(logging.WARNING, logger="mlframe.training.core._phase_train_one_target")
    _resolve_weight_schemas_and_warn_val_placement({"recency": [1.0]}, split_config, ctx)
    assert ctx._val_placement_warn_emitted is False
