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


def _read_main_or_split() -> str:
    """The ``train_mlframe_models_suite`` body was carved out of ``main.py``
    into ``_main_train_suite.py`` -> further into ``_main_train_suite_phases.py``
    during successive monolith-split waves; the deepcopy assignments moved
    with each carve. Concat all known carve siblings so the source-grep
    boundary check still matches the relocated code regardless of which
    carve generation owns it."""
    import pathlib
    import mlframe as _mlframe

    _core = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core"
    primary = (_core / "main.py").read_text(encoding="utf-8")
    for _sibname in ("_main_train_suite.py", "_main_train_suite_phases.py"):
        sib = _core / _sibname
        if sib.exists():
            primary = primary + "\n" + sib.read_text(encoding="utf-8")
    return primary


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
    """Slug maps stored on metadata must be dict() copies, not ctx aliases.
    Long-running serving process: each predict's slug-fallback setdefault would
    otherwise mutate the loaded metadata in place -> phantom slugs accumulate
    across the session.

    ``_setup_helpers.py`` was carved into themed siblings; the metadata
    finaliser that stores the slug maps moved to ``_setup_helpers_metadata.py``.
    Concat parent + sibling so the source-grep guard survives the split.
    """
    import pathlib
    import mlframe as _mlframe

    _core = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core"
    src = (_core / "_setup_helpers.py").read_text(encoding="utf-8")
    sib = _core / "_setup_helpers_metadata.py"
    if sib.exists():
        src += "\n" + sib.read_text(encoding="utf-8")
    assert "dict(ctx.slug_to_original_target_type)" in src
    assert "dict(ctx.slug_to_original_target_name)" in src


def test_discovery_cache_payload_consumed_via_defensive_copy():
    """Cached payload list/dict consumed via list(...) / dict(...) wrapper at the
    load boundary. Prevents future LRU-sidecar regression (wave 11 #5)."""
    import pathlib
    import mlframe as _mlframe

    src = (pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core" / "_phase_composite_discovery.py").read_text(encoding="utf-8")
    assert 'list(\n                    _cached_payload.get("specs_export") or []' in src or 'list(_cached_payload.get("specs_export") or [])' in src
    assert 'dict(\n                    _cached_payload.get("filter_drops") or {}' in src or 'dict(_cached_payload.get("filter_drops") or {})' in src
