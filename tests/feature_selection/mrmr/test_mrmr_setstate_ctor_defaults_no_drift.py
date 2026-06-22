"""D5 regression (2026-06-22): __setstate__ legacy defaults must not DRIFT from the ctor.

``__setstate__`` injects defaults for attribute-less legacy pickles. Historically those defaults
were hand-written literals that could (and did -- ``cluster_aggregate_mode``) silently drift from
the constructor default, so an old pickle resurrected to a DIFFERENT configuration than a freshly
constructed estimator. The fix derives every shared ctor-param default from the single source of
truth (``_ctor_defaults()`` reading ``__init__``'s signature), exempting only an explicit, documented
``_SETSTATE_LEGACY_OVERRIDES`` allowlist of intentional legacy-byte-equivalence divergences.

These tests pin that invariant: a legacy-injected ctor-param default ALWAYS equals the ctor default
unless the key is on the documented override allowlist.
"""
from __future__ import annotations

import ast
import inspect
import pickle
import textwrap

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _literal_setstate_defaults() -> dict:
    """The literal ``defaults = {...}`` dict spelled out in ``__setstate__`` (pre-overlay)."""
    src = textwrap.dedent(inspect.getsource(MRMR.__setstate__))
    fn = ast.parse(src).body[0]
    out = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "defaults" for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                try:
                    out[ast.literal_eval(k)] = ast.literal_eval(v)
                except Exception:
                    out[ast.literal_eval(k)] = "<EXPR>"
    return out


def test_shared_ctor_defaults_have_no_drift():
    ctor = MRMR._ctor_defaults()
    lit = _literal_setstate_defaults()
    overrides = set(MRMR._SETSTATE_LEGACY_OVERRIDES)
    drift = {}
    for k, v in lit.items():
        if k in ctor and k not in overrides and v != "<EXPR>":
            if v != ctor[k]:
                drift[k] = (v, ctor[k])
    assert not drift, (
        "setstate legacy defaults drifted from the ctor (add to _SETSTATE_LEGACY_OVERRIDES "
        f"only if the divergence is intentional): {drift}"
    )


def test_override_allowlist_is_actually_divergent():
    """Every key on the override allowlist MUST genuinely differ from the ctor default --
    otherwise it is dead allowlist noise hiding a key that should be auto-sourced."""
    ctor = MRMR._ctor_defaults()
    lit = _literal_setstate_defaults()
    for k in MRMR._SETSTATE_LEGACY_OVERRIDES:
        assert k in ctor, f"override key {k!r} is not a constructor parameter"
        assert lit.get(k) != ctor[k], (
            f"override key {k!r} no longer diverges from the ctor (lit={lit.get(k)!r}, "
            f"ctor={ctor[k]!r}); remove it from _SETSTATE_LEGACY_OVERRIDES so it auto-sources"
        )


def test_legacy_empty_pickle_matches_ctor_for_shared_keys():
    """An attribute-less legacy pickle, refit, must equal a fresh ctor for every shared,
    non-override key."""
    fresh = MRMR()
    ctor = MRMR._ctor_defaults()
    overrides = set(MRMR._SETSTATE_LEGACY_OVERRIDES)
    lit = _literal_setstate_defaults()

    m = MRMR.__new__(MRMR)
    m.__setstate__({})  # empty legacy state -> all defaults injected
    for k in lit:
        if k in ctor and k not in overrides:
            assert getattr(m, k) == getattr(fresh, k), (
                f"legacy-refit {k!r}={getattr(m, k)!r} != fresh {getattr(fresh, k)!r}"
            )


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
            assert getattr(back, k) == getattr(est, k) or (
                isinstance(getattr(est, k), float) and np.isnan(getattr(est, k))
            ), f"pickle round-trip changed {k!r}"
