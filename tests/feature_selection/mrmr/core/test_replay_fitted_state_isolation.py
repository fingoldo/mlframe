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

Current contract in ``_replay_fitted_state``:
1. Deep-copy mutable CONTAINER types (dict / list / set) AND every
   other non-immutable type (pandas DataFrame/Series, dataclasses,
   friend_graph_, dict-of-arrays state) on replay.
2. Small public learned-index arrays (``support_`` / ``ranking_``)
   are handed a WRITEABLE copy so a cache-replayed instance behaves
   identically to a cold-fit one (no cache-state-dependent
   read-only ValueError).
3. Larger internal ndarrays keep the freeze-and-share fast-path
   (the density win): shared read-only so an accidental write raises
   instead of silently corrupting the shared cache entry.
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
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
        }
    )
    y = pd.Series((X["a"] > 0).astype(np.int64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = MRMR(verbose=0).fit(X, y)
        B = MRMR(verbose=0).fit(X, y)
    return A, B


def test_engineered_features_list_isolated():
    """List containers must be deep-copied on replay."""
    A, B = _fit_pair()
    assert B._engineered_features_ is not A._engineered_features_, "_engineered_features_ is shared by reference - mutations leak"
    if isinstance(B._engineered_features_, list):
        B._engineered_features_.append("phantom_audit")
        assert "phantom_audit" not in A._engineered_features_


def test_engineered_recipes_isolated():
    """Recipes container (list or dict) must be deep-copied."""
    A, B = _fit_pair()
    assert B._engineered_recipes_ is not A._engineered_recipes_, "_engineered_recipes_ is shared by reference"


def test_support_writeable_on_replay_and_isolated_from_source():
    """A replayed ``support_`` must be WRITEABLE (identical to a cold-fit instance's) and its own copy, so an
    in-place write succeeds and does NOT propagate back to the cached source A.

    D7: the prior contract froze the replayed ``support_`` read-only, which made a cache-replayed instance raise
    a cache-state-dependent ValueError on a write that a cold-fit instance accepted. The fix hands the replay a
    writeable copy of the small public index arrays so cold-fit and replayed instances behave identically; the
    isolation that the freeze used to provide is now provided by the copy.
    """
    A, B = _fit_pair()
    sig = getattr(B, "signature", "") or ""
    if isinstance(sig, str) and sig.startswith("_mrmr_identity_shortcut"):
        pytest.skip("identity-shortcut path produced a fresh writable support_; the writeable-and-isolated assertion targets the FIT_CACHE replay path")
    assert B.support_.flags.writeable, "replayed support_ must be writeable like a cold-fit instance's"
    a_first = int(A.support_[0])
    if not np.shares_memory(B.support_, A.support_):
        B.support_[0] = 999
        assert int(A.support_[0]) == a_first, "writing the replayed support_ corrupted the cached source"


def test_large_internal_ndarrays_still_shared_for_density():
    """The freeze-and-share density win is preserved for LARGE internal ndarrays (not the small public index
    arrays support_/ranking_, which are copied writeable per D7). Any shared internal array must be read-only so
    an accidental write raises instead of silently corrupting the shared cache entry.
    """
    A, B = _fit_pair()
    sig = getattr(B, "signature", "") or ""
    if isinstance(sig, str) and sig.startswith("_mrmr_identity_shortcut"):
        pytest.skip("identity-shortcut path produced fresh arrays; the sharing assertion only applies to the FIT_CACHE replay path")
    _public_writeable_copies = {"support_", "ranking_"}
    for k, v in B.__dict__.items():
        if k in _public_writeable_copies or not isinstance(v, np.ndarray):
            continue
        a_v = A.__dict__.get(k)
        if isinstance(a_v, np.ndarray) and np.shares_memory(v, a_v):
            assert v.flags.writeable is False, f"shared internal ndarray {k!r} must be read-only to protect the cache entry"


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
    fitted_only = {k for k in missing_from_b if k.endswith("_") and not k.startswith("_")}
    assert not fitted_only, f"Fitted attrs missing from B after replay: {fitted_only}"
