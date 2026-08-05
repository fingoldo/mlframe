"""TRAINING_CORE_B-1 (2026-08-05 audit): the identity-equivalent-pre_pipeline dedup 'break' used to exit
the weight loop BEFORE ``_build_and_record_model_schema`` / ``pipeline_cache.set`` ran for the model that
JUST trained and triggered the dedup detection -- silently disabling predict-time schema-drift hard-fail
protection for that specific model (mirrored in both ``_phase_train_one_target_body.py`` and
``_phase_train_one_target_weight_iteration.py``). Fixed by moving the identity-equivalent check to AFTER
those two calls. This test drives ``_run_one_weight_iteration`` directly (the smaller, more directly
callable of the two sibling call sites) with ``process_model`` stubbed to a fast fake and the pre_pipeline
marked identity-equivalent, and asserts both side effects fired before the break signal is returned.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import mlframe.training.core._phase_train_one_target_weight_iteration as wi


def test_identity_equivalent_dedup_still_records_schema_and_caches_before_break(monkeypatch):
    """The model that triggers the identity-equivalent dedup must still get its schema recorded and
    its transformed frames cached before the weight-loop break signal is returned."""
    calls = []

    def _fake_process_model(**kwargs):
        """Stand in for a real model fit: returns immediately and marks pre_pipeline identity-equivalent."""
        pp = kwargs["pre_pipeline"]
        pp._mlframe_identity_equivalent = True
        return (
            {"n_features": 3},  # trainset_features_stats
            pp,  # pre_pipeline
            np.zeros((5, 3)),  # train_df_transformed
            None,  # val_df_transformed
            None,  # test_df_transformed
        )

    def _fake_build_and_record_model_schema(**kwargs):
        """Spy: record that schema-building was invoked."""
        calls.append("schema")

    def _fake_run_per_model_post_train_tail(**kwargs):
        """No-op stand-in for the post-train tail."""

    monkeypatch.setattr(wi, "process_model", _fake_process_model)
    monkeypatch.setattr(wi, "_build_and_record_model_schema", _fake_build_and_record_model_schema)
    monkeypatch.setattr(wi, "_run_per_model_post_train_tail", _fake_run_per_model_post_train_tail)
    monkeypatch.setattr(wi, "_build_process_model_kwargs", lambda **kwargs: {"pre_pipeline": kwargs["pre_pipeline"]})
    monkeypatch.setattr(wi, "_clone_model_with_sticky_flags", lambda **kwargs: (MagicMock(), kwargs["_ngb_fallback_snapshot"]))
    monkeypatch.setattr(wi, "is_neural_model", lambda name: False)

    pipeline_cache = SimpleNamespace(get=lambda key: None, set=lambda *a, **k: calls.append("cache"))
    pre_pipeline = SimpleNamespace(_mlframe_identity_equivalent=False)
    target_type = SimpleNamespace(is_multi_target_regression=False)
    behavior_config = SimpleNamespace(mlp_drop_per_group_constants=False, model_file_hash_suffix=False, continue_on_model_failure=False)
    feature_selection_config = SimpleNamespace(skip_identity_equivalent_pre_pipelines=True)

    result = wi._run_one_weight_iteration(
        ctx=SimpleNamespace(),
        weight_name="uniform",
        weight_values=None,
        common_params={"model_name": "ridge"},
        mlframe_model_name="ridge",
        polars_fastpath_active=False,
        prepared_train=None,
        prepared_val=None,
        prepared_test=None,
        tier_pandas={"train_df": "df_train"},
        behavior_config=behavior_config,
        cur_target_name="t",
        cur_target_values=None,
        _schema_hash="hash123",
        _input_schema={},
        _mlp_extreme_ar_fired=False,
        model_file="model.pkl",
        target_type=target_type,
        pre_pipeline=pre_pipeline,
        pre_pipeline_name="mrmr",  # a real (non-ordinary) FS pipeline name so _pp_name_stripped is truthy
        models={},
        _model_entry="ridge",
        models_params={"ridge": {"model": MagicMock(), "fit_params": {}}},
        trainset_features_stats={},
        verbose=False,
        pipeline_cache=pipeline_cache,
        cache_key="cache_key_1",
        polars_pipeline_applied=False,
        strategy=SimpleNamespace(supports_polars=False),
        metadata={},
        _non_neural_train_times=[],
        test_df_pd=None,
        current_test_target=None,
        _train_idx=None,
        _cached_init_params=None,
        _forward_dataset_reuse_cache=lambda *a, **k: None,
        _build_feature_selection_report=None,
        _selector_params_hash=None,
        _unwrap_selector=None,
        ens_models=[],
        _model_idx_in_run=1,
        _total_models_in_run=2,
        _pp_name_stripped="mrmr",
        use_ordinary_models=True,
        feature_selection_config=feature_selection_config,
        _cb_extra_fit_invariant=None,
        _neural_extra_fit_invariant=None,
        _ngb_fallback_snapshot=None,
    )

    assert result["break_model_loop"] is True, "the dedup break signal must still fire"
    assert "schema" in calls, "the model that triggered dedup must still have its schema recorded"
    assert "cache" in calls, "the model that triggered dedup must still have its transformed frames cached"
