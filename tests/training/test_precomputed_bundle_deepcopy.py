"""Sensor: precomputed bundle slots assigned into metadata via deepcopy, not alias.

Pre-fix shape (wave 11 #3): main.py did
``metadata["composite_target_specs"] = precomputed.composite_target_specs`` as a
shared reference. If a downstream phase ever did ``setdefault`` /
``[key] =`` on metadata's slot, the caller's precomputed bundle was mutated in
place and the change resurfaced in the next suite call reusing the same bundle.

Same for ``precomputed.dummy_baselines``.

Post-fix: ``copy.deepcopy`` at the assignment site decouples the caller's bundle.
"""

from __future__ import annotations


def test_precomputed_composite_specs_decoupled_from_metadata_slot():
    """Mutating ``metadata['composite_target_specs']`` after the precomputed
    branch fires must NOT mutate the caller's precomputed bundle -- the deepcopy
    decouples them (wave 11 #3 regression: shared alias leaked across suite calls)."""
    from types import SimpleNamespace
    from mlframe.training.core._main_train_suite_phases import (
        maybe_apply_composite_target_specs_precomputed,
    )

    bundle = {"targ_A": {"recipe": ["a", "b"]}}
    precomputed = SimpleNamespace(composite_target_specs=bundle, dummy_baselines={})
    metadata: dict = {}
    fired = maybe_apply_composite_target_specs_precomputed(
        _precomp_fp_ok=True,
        precomputed=precomputed,
        metadata=metadata,
        verbose=0,
    )
    assert fired is True
    assert metadata["composite_target_specs"] == bundle
    # Downstream mutation of the metadata slot.
    metadata["composite_target_specs"]["targ_LATE"] = {"recipe": ["x"]}
    metadata["composite_target_specs"]["targ_A"]["recipe"].append("MUT")
    # Caller's bundle stays pristine.
    assert "targ_LATE" not in bundle
    assert bundle["targ_A"]["recipe"] == ["a", "b"]


def test_precomputed_dummy_baselines_decoupled_from_metadata_slot():
    """Precomputed dummy baselines decoupled from metadata slot."""
    from types import SimpleNamespace
    from mlframe.training.core._main_train_suite_phases import (
        maybe_apply_dummy_baselines_precomputed,
    )

    bundle = {"targ_A": {"rmse": [1.0, 2.0]}}
    precomputed = SimpleNamespace(composite_target_specs={}, dummy_baselines=bundle)

    class _Cfg:
        """Groups tests covering cfg."""
        enabled = True

        def model_copy(self, update):
            """Model copy."""
            new = _Cfg()
            new.enabled = update["enabled"]
            return new

    ctx = SimpleNamespace()
    metadata: dict = {}
    cfg_out = maybe_apply_dummy_baselines_precomputed(
        _precomp_fp_ok=True,
        precomputed=precomputed,
        metadata=metadata,
        dummy_baselines_config=_Cfg(),
        ctx=ctx,
        verbose=0,
    )
    # Per-target compute is short-circuited.
    assert cfg_out.enabled is False
    assert metadata["dummy_baselines"] == bundle
    metadata["dummy_baselines"]["targ_A"]["rmse"].append(99.0)
    metadata["dummy_baselines"]["targ_LATE"] = {"rmse": [0.0]}
    assert "targ_LATE" not in bundle
    assert bundle["targ_A"]["rmse"] == [1.0, 2.0]


def test_setup_helpers_slug_maps_dict_copy():
    """Slug maps stored on metadata are copies, not ctx aliases.
    Long-running serving process: each predict's slug-fallback setdefault would
    otherwise mutate the loaded metadata in place -> phantom slugs accumulate
    across the session.
    """
    from types import SimpleNamespace

    from mlframe.training.core import _finalize_and_save_metadata

    ctx = SimpleNamespace(
        metadata={"model_name": "m", "target_name": "t", "mlframe_models": []},
        outlier_detector=None,
        outlier_detection_result={},
        trainset_features_stats=None,
        slug_to_original_target_type={"reg": "Regression"},
        slug_to_original_target_name={"t": "Target"},
        data_dir="",
        models_dir="",
        target_name="t",
        model_name="m",
        verbose=0,
    )
    _finalize_and_save_metadata(ctx)
    assert ctx.metadata["slug_to_original_target_type"] == {"reg": "Regression"}
    assert ctx.metadata["slug_to_original_target_name"] == {"t": "Target"}
    assert ctx.metadata["slug_to_original_target_type"] is not ctx.slug_to_original_target_type
    assert ctx.metadata["slug_to_original_target_name"] is not ctx.slug_to_original_target_name
    ctx.metadata["slug_to_original_target_type"]["phantom"] = "x"
    ctx.metadata["slug_to_original_target_name"]["phantom"] = "x"
    assert ctx.slug_to_original_target_type == {"reg": "Regression"}
    assert ctx.slug_to_original_target_name == {"t": "Target"}


def test_discovery_cache_payload_consumed_via_defensive_copy():
    """Cached payload list/dict consumed through fresh containers at the load boundary.
    Prevents future LRU-sidecar regression (wave 11 #5)."""
    from mlframe.training.core._phase_composite_discovery import _replay_cached_payload_into_metadata

    payload = {"specs_export": [{"name": "s1"}], "failures": [{"name": "f1"}], "filter_drops": {"a": 1}}
    metadata: dict = {"composite_target_specs": {}, "composite_target_failures": {}}
    _replay_cached_payload_into_metadata(metadata, "REGRESSION", "y", payload)
    assert metadata["composite_target_specs"]["REGRESSION"]["y"] == payload["specs_export"]
    assert metadata["composite_target_failures"]["REGRESSION"]["y"] == payload["failures"]
    assert metadata["composite_target_filter_drops"]["REGRESSION"]["y"] == payload["filter_drops"]
    metadata["composite_target_specs"]["REGRESSION"]["y"].append({"name": "late"})
    metadata["composite_target_failures"]["REGRESSION"]["y"].clear()
    metadata["composite_target_filter_drops"]["REGRESSION"]["y"]["b"] = 2
    assert payload == {"specs_export": [{"name": "s1"}], "failures": [{"name": "f1"}], "filter_drops": {"a": 1}}

    empty: dict = {"composite_target_specs": {}, "composite_target_failures": {}}
    _replay_cached_payload_into_metadata(empty, "REGRESSION", "y", {})
    assert empty["composite_target_specs"]["REGRESSION"]["y"] == []
    assert empty["composite_target_filter_drops"]["REGRESSION"]["y"] == {}
