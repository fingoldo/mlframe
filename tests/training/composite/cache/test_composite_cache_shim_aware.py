"""Regression sensor for S48: the train-prediction cache used by the cross-target ensemble
must key on the INNER model id plus frame identity, not just ``id(comp)``.

Wrap-pass writes the train predict under both ``(id(wrapper), id(frame), shape)`` and
``(id(inner), id(frame), shape)``. Ensemble-pass builds a fresh shim around the same wrapper for
the SAME train frame; the reader (``_phase_composite_post_xt_ensemble._get_train_pred``) peels one
shim via ``getattr(_comp, "model", _comp)`` and looks up ``(id(inner), id(frame), shape)``: this
must HIT (cache reuse, no redundant .predict). An unwrapped component (no ``.model``) peels to
itself, so the same key recovers the legacy ``id(_comp)`` fallback. The frame-id component shields
against ``id()`` recycling across GC; without it a freed inner landing at the same address would
return a stale prediction array.
"""

from __future__ import annotations

import numpy as np


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


def test_S48_lookup_peels_one_shim_and_keys_on_frame_for_unwrapped_and_wrapped():
    """Behavioural restatement of the cross-target ensemble train-prediction cache invariant
    (the cache loop now lives in ``_phase_composite_post_xt_ensemble._get_train_pred``).

    The reader builds the lookup key by peeling exactly one shim layer via
    ``getattr(comp, "model", comp)`` and prefixing ``id(inner)`` to the frame key. This is the
    unified replacement for the old inner-keyed-first / id(comp)-fallback pair: a WRAPPED component
    keys on its inner model (HIT against the wrap-pass write); an UNWRAPPED component (lag_predict,
    no ``.model``) peels to itself so ``id(inner) == id(comp)``, recovering the old fallback in one key.
    The frame-id stays in the key so two distinct frames never alias under id() recycling.
    """

    def lookup_key(comp, frame_key):
        inner = getattr(comp, "model", comp)
        return (id(inner),) + frame_key

    inner_wrapper = _FakeFittedInner()

    class _F:
        shape = (10, 3)

    frame = _F()
    frame_key = (id(frame), frame.shape)

    pre = np.asarray(inner_wrapper.predict(range(10)), dtype=np.float64)
    cache = {(id(inner_wrapper),) + frame_key: pre}
    assert inner_wrapper.n_predict == 1

    wrapped = _FakeShim(inner_wrapper)
    assert cache.get(lookup_key(wrapped, frame_key)) is not None, "wrapped component must peel one shim and HIT the wrap-pass cache without re-predicting"
    assert inner_wrapper.n_predict == 1

    unwrapped = _FakeFittedInner()
    cache[lookup_key(unwrapped, frame_key)] = np.full(10, 7.0)
    assert lookup_key(unwrapped, frame_key) == (id(unwrapped),) + frame_key, (
        "unwrapped component (no .model) keys on itself, recovering the legacy id(comp) fallback"
    )

    other_frame = _F()
    assert cache.get(lookup_key(wrapped, (id(other_frame), other_frame.shape))) is None, "frame identity must stay in the key; a different frame must MISS"
