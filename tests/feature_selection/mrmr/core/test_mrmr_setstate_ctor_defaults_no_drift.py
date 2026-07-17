"""D5 regression (2026-06-22): __setstate__ legacy defaults must not DRIFT from the ctor.

``__setstate__`` injects defaults for attribute-less legacy pickles. Historically those defaults
were hand-written literals that could (and did -- ``cluster_aggregate_mode``) silently drift from
the constructor default, so an old pickle resurrected to a DIFFERENT configuration than a freshly
constructed estimator. The fix derives every shared ctor-param default from the single source of
truth (``_ctor_defaults()`` reading ``__init__``'s signature), exempting only an explicit, documented
``_SETSTATE_LEGACY_OVERRIDES`` allowlist of intentional legacy-byte-equivalence divergences.

These tests pin that invariant BEHAVIOURALLY (no ``inspect.getsource`` source-text inspection,
per feedback_behavioral_tests): they resurrect an attribute-less legacy pickle via
``__setstate__({})`` and compare the resulting live attributes against a fresh ctor. After the D5
overlay, an attribute-less pickle's injected value for a shared key equals the ctor default UNLESS
the key is on ``_SETSTATE_LEGACY_OVERRIDES`` (which keeps its legacy literal). So the live
post-setstate attributes ARE the effective legacy defaults -- the exact thing we need to assert.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _legacy_setstate_attrs() -> MRMR:
    """An attribute-less legacy pickle resurrected: every injected default is now a live attr."""
    m = MRMR.__new__(MRMR)
    m.__setstate__({})  # empty legacy state -> all defaults injected (+ D5 ctor-overlay applied)
    return m


def test_shared_ctor_defaults_have_no_drift():
    """For every shared, non-override ctor param, the resurrected legacy default == ctor default."""
    fresh = MRMR()
    m = _legacy_setstate_attrs()
    ctor = MRMR._ctor_defaults()
    overrides = set(MRMR._SETSTATE_LEGACY_OVERRIDES)
    drift = {}
    for k in ctor:
        if k in overrides or not hasattr(m, k):
            continue
        if getattr(m, k) != getattr(fresh, k):
            drift[k] = (getattr(m, k), getattr(fresh, k))
    assert not drift, f"setstate legacy defaults drifted from the ctor (add to _SETSTATE_LEGACY_OVERRIDES only if the divergence is intentional): {drift}"


def test_override_allowlist_is_actually_divergent():
    """Every key on the override allowlist MUST genuinely differ from the ctor default --
    otherwise it is dead allowlist noise hiding a key that should be auto-sourced.

    The overlay SKIPS override keys, so the resurrected legacy attr keeps its legacy literal;
    comparing it to the ctor default proves the divergence is real."""
    m = _legacy_setstate_attrs()
    ctor = MRMR._ctor_defaults()
    for k in MRMR._SETSTATE_LEGACY_OVERRIDES:
        assert k in ctor, f"override key {k!r} is not a constructor parameter"
        assert hasattr(m, k), f"override key {k!r} was not injected by __setstate__"
        assert getattr(m, k) != ctor[k], (
            f"override key {k!r} no longer diverges from the ctor (legacy={getattr(m, k)!r}, "
            f"ctor={ctor[k]!r}); remove it from _SETSTATE_LEGACY_OVERRIDES so it auto-sources"
        )


def test_legacy_empty_pickle_matches_ctor_for_shared_keys():
    """An attribute-less legacy pickle, refit, must equal a fresh ctor for every shared,
    non-override key."""
    fresh = MRMR()
    ctor = MRMR._ctor_defaults()
    overrides = set(MRMR._SETSTATE_LEGACY_OVERRIDES)
    m = _legacy_setstate_attrs()
    for k in ctor:
        if k in overrides or not hasattr(m, k):
            continue
        assert getattr(m, k) == getattr(fresh, k), f"legacy-refit {k!r}={getattr(m, k)!r} != fresh {getattr(fresh, k)!r}"


def test_named_drift_key_cluster_aggregate_mode():
    # The originally-drifted key: legacy refit must match the corrected ctor default.
    assert MRMR().cluster_aggregate_mode == "replace"
    m = MRMR.__new__(MRMR)
    m.__setstate__({})
    assert m.cluster_aggregate_mode == "replace"


def test_full_pickle_round_trip_preserves_params():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + X["f1"] > 0).astype(int).to_numpy()
    est = MRMR(max_runtime_mins=0.05)
    est.fit(X, y)
    blob = pickle.dumps(est)
    back = pickle.loads(blob)
    for k in MRMR._ctor_defaults():
        if hasattr(est, k):
            assert getattr(back, k) == getattr(est, k) or (isinstance(getattr(est, k), float) and np.isnan(getattr(est, k))), f"pickle round-trip changed {k!r}"
