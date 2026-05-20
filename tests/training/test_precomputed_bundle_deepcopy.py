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

import copy

import pytest


def test_main_py_uses_deepcopy_for_precomputed_composite_specs():
    """Source-level guard: the deepcopy must remain at the assignment site."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "core" / "main.py"
    ).read_text(encoding="utf-8")
    assert "_copy.deepcopy(precomputed.composite_target_specs)" in src, (
        "main.py must use deepcopy when assigning precomputed.composite_target_specs to "
        "metadata. Without this, a downstream phase mutating the metadata slot also "
        "mutates the caller's precomputed bundle -- the leak resurfaces in the next "
        "suite call reusing the same bundle (wave 11 #3 regression)."
    )


def test_main_py_uses_deepcopy_for_precomputed_dummy_baselines():
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "core" / "main.py"
    ).read_text(encoding="utf-8")
    assert "_copy_db.deepcopy(precomputed.dummy_baselines)" in src, (
        "main.py must use deepcopy when assigning precomputed.dummy_baselines (wave 11 #3)."
    )


def test_setup_helpers_slug_maps_dict_copy():
    """Slug maps stored on metadata must be dict() copies, not ctx aliases.
    Long-running serving process: each predict's slug-fallback setdefault would
    otherwise mutate the loaded metadata in place -> phantom slugs accumulate
    across the session."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "core" / "_setup_helpers.py"
    ).read_text(encoding="utf-8")
    assert "dict(ctx.slug_to_original_target_type)" in src
    assert "dict(ctx.slug_to_original_target_name)" in src


def test_discovery_cache_payload_consumed_via_defensive_copy():
    """Cached payload list/dict consumed via list(...) / dict(...) wrapper at the
    load boundary. Prevents future LRU-sidecar regression (wave 11 #5)."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "core" / "_phase_composite_discovery.py"
    ).read_text(encoding="utf-8")
    assert "list(\n                    _cached_payload.get(\"specs_export\") or []" in src or \
           "list(_cached_payload.get(\"specs_export\") or [])" in src
    assert "dict(\n                    _cached_payload.get(\"filter_drops\") or {}" in src or \
           "dict(_cached_payload.get(\"filter_drops\") or {})" in src
