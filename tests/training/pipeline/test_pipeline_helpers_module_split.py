"""Wave 93 (2026-05-21): split _pipeline_helpers.py (1025 lines)
into _pipeline_helpers.py (now 772 lines) + new _pipeline_cache.py
(296 lines). The cache machinery (lock, sentinel, fingerprint,
signature, get/set/clear) moved to the sibling file; the original
re-exports the cache symbols so existing imports keep working.

Backward-compat preserved: every symbol previously importable from
``mlframe.training.pipeline._pipeline_helpers`` is still importable from there.
"""

from __future__ import annotations

from pathlib import Path


def test_cache_symbols_still_importable_from_facade() -> None:
    """Cache symbols still importable from facade."""
    from mlframe.training.pipeline._pipeline_helpers import (
        _fresh_uncachable,
        _content_fingerprint_for_cache,
        _pipeline_signature_for_cache,
        _pre_pipeline_cache_key,
        _pre_pipeline_cache_get,
        _pre_pipeline_cache_set,
        _pre_pipeline_cache_clear,
        _PRE_PIPELINE_CACHE_MAX,
    )

    assert callable(_fresh_uncachable)
    assert callable(_content_fingerprint_for_cache)
    assert callable(_pipeline_signature_for_cache)
    assert callable(_pre_pipeline_cache_key)
    assert callable(_pre_pipeline_cache_get)
    assert callable(_pre_pipeline_cache_set)
    assert callable(_pre_pipeline_cache_clear)
    assert isinstance(_PRE_PIPELINE_CACHE_MAX, int)


def test_pipeline_ops_symbols_still_importable_from_facade() -> None:
    """Pipeline ops symbols still importable from facade."""
    from mlframe.training.pipeline._pipeline_helpers import (
        _prepare_test_split,
        _extract_feature_selector,
        _is_fitted,
        _multilabel_target_to_1d_for_supervised_encoders,
        _passthrough_cols_fit_transform,
        _apply_pre_pipeline_transforms,
    )

    for fn in (
        _prepare_test_split,
        _extract_feature_selector,
        _is_fitted,
        _multilabel_target_to_1d_for_supervised_encoders,
        _passthrough_cols_fit_transform,
        _apply_pre_pipeline_transforms,
    ):
        assert callable(fn), fn


def test_facade_below_1k_line_threshold() -> None:
    """Facade below 1k line threshold."""
    root = Path(__file__).resolve().parent.parent.parent.parent / "src" / "mlframe" / "training" / "pipeline"
    facade = root / "_pipeline_helpers.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"_pipeline_helpers.py is {n} lines, still over the 1k threshold"


def test_cache_module_owns_the_moved_symbols() -> None:
    """Identity: the facade and the cache module expose the SAME object."""
    from mlframe.training.pipeline import _pipeline_helpers, _pipeline_cache

    for name in (
        "_UncachableSentinel",
        "_fresh_uncachable",
        "_content_fingerprint_for_cache",
        "_pipeline_signature_for_cache",
        "_pre_pipeline_cache_key",
        "_pre_pipeline_cache_get",
        "_pre_pipeline_cache_set",
        "_pre_pipeline_cache_clear",
    ):
        assert getattr(_pipeline_helpers, name) is getattr(_pipeline_cache, name), name


def test_content_fingerprint_round_trips_via_facade() -> None:
    """Functional smoke: fingerprint two identical frames -> identical key."""
    import pandas as pd
    from mlframe.training.pipeline._pipeline_helpers import _content_fingerprint_for_cache

    df_a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    df_b = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    df_c = pd.DataFrame({"x": [1.0, 2.0, 4.0]})
    assert _content_fingerprint_for_cache(df_a) == _content_fingerprint_for_cache(df_b)
    assert _content_fingerprint_for_cache(df_a) != _content_fingerprint_for_cache(df_c)


def test_cache_get_set_clear_round_trip() -> None:
    """The LRU dict works after the split: set -> get returns the same value; clear empties."""
    import pandas as pd
    from mlframe.training.pipeline._pipeline_helpers import (
        _pre_pipeline_cache_set,
        _pre_pipeline_cache_get,
        _pre_pipeline_cache_clear,
    )

    _pre_pipeline_cache_clear()
    train = pd.DataFrame({"x": [1.0, 2.0]})
    val = pd.DataFrame({"x": [3.0]})
    pipeline = object()  # not strictly an sklearn pipeline; signature-fallback path
    train_out = pd.DataFrame({"x": [1.0, 2.0]})
    val_out = pd.DataFrame({"x": [3.0]})
    _pre_pipeline_cache_set(train, val, pipeline, train_out, val_out, target_name="t")
    hit = _pre_pipeline_cache_get(train, val, pipeline, target_name="t")
    assert hit is not None
    assert hit[0] is train_out and hit[1] is val_out
    _pre_pipeline_cache_clear()
    miss = _pre_pipeline_cache_get(train, val, pipeline, target_name="t")
    assert miss is None
