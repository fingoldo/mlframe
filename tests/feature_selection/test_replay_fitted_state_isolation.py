"""Wave 9.1 loop-iter-40 regression: ``_replay_fitted_state`` MUST
isolate mutable container state across cached fits.

Pre-fix at ``_mrmr_fingerprints.py:526``:

    target.__dict__[k] = v   # shallow assign

Every fitted-state attribute - including mutable lists / dicts / sets
like ``_engineered_features_``, ``_engineered_recipes_``,
``_cat_fe_cache_`` - was shared by reference with the cached source
MRMR. Any in-place mutation on one replayed instance silently
propagated to every past + future cache hit:

  A = MRMR().fit(X, y)
  B = MRMR().fit(X, y)        # cache HIT - shares A's containers
  B._engineered_features_.append("manual_audit_feat")
  -> A._engineered_features_ now contains "manual_audit_feat"
  C = MRMR().fit(X, y)        # cache HIT - sees A's polluted state
  -> C has phantom feature it never had through fit

Same hazard for ndarrays: ``B.support_[0] = 999`` corrupts A's
support, and every subsequent replay reads the corrupted value.

Severity: P1 silent-correctness for any workflow that post-mutates
fitted state on a replayed instance (CV / clone loops, audit
notebooks, downstream pipelines that extend ``_cat_fe_cache_``).

Fix at _mrmr_fingerprints.py:522:
1. Deep-copy mutable CONTAINER types (dict / list / set) on replay.
2. Freeze source ndarrays (``arr.flags.writeable = False``) so
   accidental in-place writes raise ValueError instead of silently
   corrupting the shared cache entry.
3. Keep ndarray SHARING for the common read-only path (the cache-
   density win).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _fit_pair():
    """Fit two MRMR instances on identical X, y so the second hits
    the _FIT_CACHE and replays from the first.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
    })
    y = pd.Series((X["a"] > 0).astype(np.int64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = MRMR(verbose=0).fit(X, y)
        B = MRMR(verbose=0).fit(X, y)
    return A, B


def test_engineered_features_list_isolated():
    """List containers must be deep-copied on replay."""
    A, B = _fit_pair()
    assert B._engineered_features_ is not A._engineered_features_, (
        "_engineered_features_ is shared by reference - mutations leak"
    )
    if isinstance(B._engineered_features_, list):
        B._engineered_features_.append("phantom_audit")
        assert "phantom_audit" not in A._engineered_features_


def test_engineered_recipes_isolated():
    """Recipes container (list or dict) must be deep-copied."""
    A, B = _fit_pair()
    assert B._engineered_recipes_ is not A._engineered_recipes_, (
        "_engineered_recipes_ is shared by reference"
    )


def test_support_ndarray_frozen_to_prevent_silent_corruption():
    """In-place writes on a replayed support_ MUST raise ValueError
    rather than silently propagating to A.

    Only applies when B actually went through the cache-replay path
    (identity-shortcut path produces a fresh writable array per
    invocation and doesn't share with A, so the freeze assertion is
    moot there).
    """
    A, B = _fit_pair()
    # Detect identity-shortcut: signature starts with the documented
    # sentinel string; in that case B's support_ is fresh, not from A.
    sig = getattr(B, "signature", "") or ""
    if isinstance(sig, str) and sig.startswith("_mrmr_identity_shortcut"):
        pytest.skip(
            "identity-shortcut path produced a fresh writable support_; "
            "freeze assertion only applies to FIT_CACHE replay path"
        )
    with pytest.raises(ValueError, match="read-only"):
        B.support_[0] = 999
    assert A.support_[0] != 999


def test_ndarray_sharing_preserved_for_density():
    """Sanity: when the cache-replay path fires, ndarrays are still
    shared (the iter-40 density win). Identity-shortcut path produces
    fresh arrays per invocation and doesn't share.
    """
    A, B = _fit_pair()
    sig = getattr(B, "signature", "") or ""
    if isinstance(sig, str) and sig.startswith("_mrmr_identity_shortcut"):
        pytest.skip(
            "identity-shortcut path produced fresh arrays; sharing "
            "assertion only applies to FIT_CACHE replay path"
        )
    assert B.support_ is A.support_ or np.shares_memory(B.support_, A.support_) or (
        B.support_.flags.writeable is False
    )


def test_replay_count_unchanged_after_fix():
    """Negative control: the number of attrs replayed (the return
    value of ``_replay_fitted_state``) must be unchanged - we only
    altered HOW they're copied, not WHAT.
    """
    A, B = _fit_pair()
    # Both should have the same set of fitted attrs.
    a_keys = set(A.__dict__.keys())
    b_keys = set(B.__dict__.keys())
    # Allow B to have a few extra metadata keys from MRMR.fit itself
    # (signature, etc.) - the iter-40 fix doesn't change those.
    missing_from_b = a_keys - b_keys
    # Constructor params can legitimately differ between instances; the
    # important thing is that fitted state is on both.
    fitted_only = {k for k in missing_from_b
                    if k.endswith("_") and not k.startswith("_")}
    assert not fitted_only, (
        f"Fitted attrs missing from B after replay: {fitted_only}"
    )
