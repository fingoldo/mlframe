"""Sensor: _phase_train_one_target dataset-cache carve preserves identity + facade under budget."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mlframe.training.core import _phase_train_one_target as parent
from mlframe.training.core import _phase_train_one_target_dataset_cache as dc


def test_w12b_phase_train_one_target_identity_preserved():
    """W12b phase train one target identity preserved."""
    assert parent._DATASET_REUSE_CACHE_ATTRS is dc._DATASET_REUSE_CACHE_ATTRS
    assert parent._forward_dataset_reuse_cache is dc._forward_dataset_reuse_cache
    assert parent._release_ctx_polars_frames is dc._release_ctx_polars_frames
    assert parent._ensure_ctx_artifacts is dc._ensure_ctx_artifacts
    assert parent._capture_dataset_reuse_cache is dc._capture_dataset_reuse_cache
    assert parent._restore_dataset_reuse_cache is dc._restore_dataset_reuse_cache


def test_w12b_phase_train_one_target_facade_under_budget():
    """W12b phase train one target facade under budget."""
    facade_loc = sum(1 for _ in Path(parent.__file__).open(encoding="utf-8"))
    assert facade_loc < 750, f"_phase_train_one_target.py LOC={facade_loc} exceeds 750 budget"


def test_w12b_phase_train_one_target_dataset_cache_smoke():
    # ctx-shaped object whose artifacts is a fresh empty dict.
    """W12b phase train one target dataset cache smoke."""
    ctx = SimpleNamespace(artifacts={})
    cache = parent._ensure_dataset_reuse_cache(ctx)
    assert cache == {}
    key = parent._dataset_reuse_cache_key("LightGBM", "MRMR")
    assert key == ("LightGBM", "MRMR")

    # capture / restore round-trip
    class _Tmpl:
        """Groups tests covering tmpl."""
        _cached_train_dmatrix = "sentinel"
        _cached_train_key = "k1"

    tpl = _Tmpl()
    parent._capture_dataset_reuse_cache(ctx, "LightGBM", tpl, pp_name="MRMR")
    fresh = type("F", (), {})()
    parent._restore_dataset_reuse_cache(ctx, "LightGBM", fresh, pp_name="MRMR")
    assert fresh._cached_train_dmatrix == "sentinel"
    assert fresh._cached_train_key == "k1"


def test_w12b_phase_train_one_target_forward_cache_skip_none():
    """W12b phase train one target forward cache skip none."""
    src = type("S", (), {"_cached_train_dmatrix": "x", "_cached_val_dmatrix": None})()
    dst = type("D", (), {})()
    parent._forward_dataset_reuse_cache(src, dst, skip_none=True)
    assert dst._cached_train_dmatrix == "x"
    assert not hasattr(dst, "_cached_val_dmatrix")
