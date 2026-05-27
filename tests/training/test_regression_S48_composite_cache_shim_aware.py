"""Regression sensor for S48: the train-prediction cache used by the cross-target ensemble
must key on the INNER model id plus frame identity, not just ``id(comp)``.

Wrap-pass writes the train predict under ``(id(wrapper), id(frame), shape)``. Ensemble-pass
builds a fresh ``PrePipelinePredictShim(_inner=wrapper, ...)`` around the same wrapper for the
SAME train frame. Reader peels one shim via ``getattr(_comp, "model", _comp)`` and looks up
``(id(inner), id(frame), shape)``: this must HIT (cache reuse, no redundant .predict). The
frame-id component shields against ``id()`` recycling across GC; without it a freed inner
landing at the same address would return a stale prediction array.
"""
from __future__ import annotations

import numpy as np
import pytest


class _FakeFittedInner:
    """Behaves like a fitted model: counts .predict() calls."""

    def __init__(self):
        self.n_predict = 0

    def predict(self, X):
        self.n_predict += 1
        n = len(X)
        # deterministic so equality checks below are stable
        return np.arange(n, dtype=np.float64)


class _FakeShim:
    """Stand-in for PrePipelinePredictShim: exposes .model = inner, .predict delegates."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)


def test_S48_cache_lookup_peels_one_shim_layer_via_dot_model():
    """A reader that builds a fresh shim around an already-cached wrapper must HIT cache without
    re-running .predict on the wrapper. This is the per-target ensemble's primary perf win."""
    inner_wrapper = _FakeFittedInner()
    cache: dict[tuple, np.ndarray] = {}

    # Simulate a (fake) DataFrame so shape attr exists. Just a list with shape.
    class _F:
        shape = (10, 3)

    frame = _F()
    frame_key = (id(frame), frame.shape)

    # Wrap pass: writes under (id(inner_wrapper),) + frame_key
    pre = np.asarray(inner_wrapper.predict(range(10)), dtype=np.float64)
    cache[(id(inner_wrapper),) + frame_key] = pre
    assert inner_wrapper.n_predict == 1

    # Ensemble pass: builds a fresh shim around the SAME inner_wrapper, looks up under
    # (id(getattr(shim, 'model', shim)),) + frame_key. Must HIT.
    fresh_shim = _FakeShim(inner_wrapper)
    _inner_for_cache = getattr(fresh_shim, "model", fresh_shim)
    got = cache.get((id(_inner_for_cache),) + frame_key)
    assert got is not None, "ensemble-pass lookup MUST hit the wrap-pass cache under (id(inner), frame_key)"
    np.testing.assert_array_equal(got, pre)
    assert inner_wrapper.n_predict == 1, ".predict must NOT be re-invoked when cache hits"


def test_S48_cache_key_includes_frame_identity_against_id_recycling():
    """Different frames must NOT alias in the cache even when the inner-model id matches.
    This is the defence against ``id()`` recycling across GC events."""
    inner_wrapper = _FakeFittedInner()
    cache: dict[tuple, np.ndarray] = {}

    class _F:
        def __init__(self, shape):
            self.shape = shape

    frame_a = _F((10, 3))
    frame_b = _F((10, 3))
    key_a = (id(inner_wrapper), id(frame_a), frame_a.shape)
    key_b = (id(inner_wrapper), id(frame_b), frame_b.shape)
    assert key_a != key_b, "two distinct frames must NOT share a cache key"

    # Write under frame_a; lookup under frame_b must miss.
    pre_a = np.full(10, 1.0)
    cache[key_a] = pre_a
    assert cache.get(key_b) is None, "frame_b lookup must miss; frame identity is part of the key"


def test_S48_lookup_in_source_uses_inner_keyed_first_fallback_second():
    """The lookup pattern in _phase_composite_post.py MUST try the inner-keyed cache first
    (the common path) and fall back to ``id(_comp)`` only on miss (defensive for unwrapped
    components like lag_predict).

    ``_phase_composite_post.py`` was carved into themed siblings; the
    cross-target ensemble loop that owns the train-prediction cache
    landed in ``_phase_composite_post_xt_ensemble.py``. Concat parent +
    sibling so the source-grep guard survives the split.
    """
    from pathlib import Path

    _core = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "training" / "core"
    src = (_core / "_phase_composite_post.py").read_text(encoding="utf-8")
    sib = _core / "_phase_composite_post_xt_ensemble.py"
    if sib.exists():
        src += "\n" + sib.read_text(encoding="utf-8")
    # Inner-keyed lookup must be the FIRST attempt; the id(_comp) fallback must come SECOND.
    inner_idx = src.find("_train_pred_cache.get((id(_inner_for_cache),) + _frame_key")
    comp_idx = src.find("_train_pred_cache.get((id(_comp),) + _frame_key")
    assert inner_idx > 0, "expected inner-keyed cache lookup (id(_inner_for_cache),) + _frame_key"
    assert comp_idx > 0, "expected id(_comp) fallback lookup"
    assert inner_idx < comp_idx, (
        "inner-keyed lookup must precede the id(_comp) fallback; "
        "fallback is for unwrapped components only."
    )
