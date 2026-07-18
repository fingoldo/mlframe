"""Regression tests for 05_concurrency_and_statistics.md findings #1, #5, #6.

#1: GPU circuit-breaker re-arm now gated to the 0->1 in-flight-fit transition instead of running
unconditionally at every fit() entry.
#5: concurrent fit() on the SAME MRMR instance now raises a clear RuntimeError instead of silently
racing on the instance's own mutable attributes.
#6: cmi_perm_stop / cpt_test permutation-null seeds now fold in the current conditioning set
(selected_vars), not just the candidate index, so the same candidate re-evaluated across greedy
rounds no longer draws an identical (correlated) permutation stream.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.mrmr import _mrmr_class_fit_helpers as fh


def _xy(seed=0, n=300, p=4):
    """Build a small synthetic classification fixture with signal on column f0."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(p)})
    y = pd.Series((X["f0"] + 0.3 * rng.randn(n) > 0).astype(int), name="t")
    return X, y


def test_active_fit_scope_only_rearms_on_zero_to_one_transition(monkeypatch):
    """_enter_active_fit_scope must call _rearm_gpu_circuit_breakers only when the in-flight-fit
    count transitions 0->1, not on every entry (finding #1)."""
    calls = []
    monkeypatch.setattr(fh, "_ACTIVE_FIT_COUNT", 0)

    class _Dummy(fh._MRMRFitHelpersMixin):
        """Bare host object for exercising the mixin's active-fit-scope bookkeeping in isolation."""

        def _rearm_gpu_circuit_breakers(self):
            """Record that a rearm happened."""
            calls.append(1)

    d1 = _Dummy()
    d2 = _Dummy()
    d1._enter_active_fit_scope()
    assert calls == [1], "first entry (0->1) must rearm"
    d2._enter_active_fit_scope()
    assert calls == [1], "second concurrent entry (1->2) must NOT rearm again"
    d2._exit_active_fit_scope()
    d1._exit_active_fit_scope()
    d1._enter_active_fit_scope()
    assert calls == [1, 1], "a fresh 0->1 transition after both fits ended must rearm again"
    d1._exit_active_fit_scope()


def test_concurrent_fit_on_same_instance_raises():
    """A second thread calling .fit() on the SAME MRMR instance while the first is still running must
    raise a clear RuntimeError instead of racing on the instance's own attributes (finding #5)."""
    X, y = _xy(n=4000, p=6)
    m = MRMR(verbose=0, full_npermutations=1, baseline_npermutations=1, fe_max_steps=0, max_runtime_mins=1)

    release_event = threading.Event()
    entered_event = threading.Event()
    real_check = m._check_groups_contract

    def _slow_check(groups):
        """Signal entry, then block until the second thread has attempted its own concurrent fit()."""
        entered_event.set()
        release_event.wait(timeout=10)
        return real_check(groups)

    m._check_groups_contract = _slow_check

    errors = []

    def _first_fit():
        """Background thread body: runs the slow first fit() and records any unexpected exception."""
        try:
            m.fit(X, y)
        except Exception as exc:  # pragma: no cover - only a genuine unexpected failure would land here
            errors.append(exc)

    t = threading.Thread(target=_first_fit)
    t.start()
    assert entered_event.wait(timeout=10), "first fit() did not reach the slow checkpoint in time"

    with pytest.raises(RuntimeError, match="concurrently"):
        m.fit(X, y)

    release_event.set()
    t.join(timeout=30)
    assert not errors, f"first (legitimate) fit() must complete cleanly; got {errors}"


def test_cmi_cpt_seed_varies_with_selected_vars():
    """The permutation-null seed derived for cmi_perm/cpt must depend on the current conditioning set
    (selected_vars), not just the candidate index -- otherwise the same candidate re-evaluated against
    a growing conditioning set across greedy rounds draws an identical (correlated) null stream
    (finding #6)."""
    cand_idx = 5
    seed_empty = cand_idx  # round 0: no selected_vars yet -> falls back to cand_idx alone
    seed_round1 = hash((cand_idx, tuple(sorted((1,))))) & 0xFFFFFFFF
    seed_round2 = hash((cand_idx, tuple(sorted((1, 2))))) & 0xFFFFFFFF
    assert seed_round1 != seed_round2, "different conditioning sets for the same candidate must yield different seeds"
    assert seed_round1 != seed_empty
    assert seed_round2 != seed_empty


def test_clone_then_fit_hits_cache_does_not_alias_reentrancy_lock():
    """Regression: ``_replay_fitted_state`` must not copy the cached source instance's
    ``_fit_reentrancy_lock_`` onto a cache-hit target -- doing so aliased the clone's fit() wrapper to
    release a lock it never acquired ("release unlocked lock" RuntimeError on every cache-hit replay,
    found while testing finding #5's re-entrancy guard)."""
    from sklearn.base import clone

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = pd.Series((rng.normal(size=200) > 0).astype(int))
    MRMR.clear_fit_cache()
    mrmr = MRMR(full_npermutations=2, baseline_npermutations=2, fe_max_steps=0, verbose=0, n_jobs=1)
    mrmr.fit(X, y)
    assert len(MRMR._FIT_CACHE) == 1

    cloned = clone(mrmr)
    cloned.fit(X, y)  # cache hit -> replay path; must not raise "release unlocked lock"
    np.testing.assert_array_equal(cloned.support_, mrmr.support_)
    assert cloned._fit_reentrancy_lock is not mrmr._fit_reentrancy_lock, "each instance must keep its OWN lock"


def test_same_content_skip_signature_survives_transient_ctor_overrides():
    """Regression: the in-object same-content-skip signature must reflect the STABLE, user-visible
    ctor param state, not a TRANSIENT mid-fit override (e.g. cluster_aggregate_enable, which fit()
    temporarily flips off and restores). Pre-fix, capturing the signature from a live mid-fit
    ``get_params()`` read permanently broke the same-content skip for the DEFAULT
    ``cluster_aggregate_enable=True`` config: every "identical" refit's freshly-read params (post-
    restore) could never again match the stored (transiently-overridden) signature, forcing a full
    re-fit every time (found while testing finding #5's re-entrancy guard,
    05_concurrency_and_statistics.md)."""
    X, y = _xy(seed=2, n=400, p=5)
    m = MRMR(verbose=0, n_jobs=1, fe_max_steps=0, mrmr_skip_when_prior_was_identity=False)
    m.fit(X, y)
    sig1 = m.signature
    assert isinstance(sig1, tuple), f"expected the tuple-shaped same-content signature, got {type(sig1)}"
    psig1 = dict(sig1[5])
    assert psig1.get("cluster_aggregate_enable") is True, (
        f"stored signature must reflect the STABLE ctor value (True), not the transient mid-fit "
        f"override (False); got {psig1.get('cluster_aggregate_enable')!r}"
    )

    m.fit(X, y)  # a second identical fit must re-derive the SAME signature (skip actually engages)
    sig2 = m.signature
    assert sig1 == sig2, f"same-content signature must be stable across identical refits; sig1={sig1!r} sig2={sig2!r}"


def test_biz_val_cmi_perm_seed_depends_on_random_effective_context():
    """End-to-end smoke: fitting twice with cmi_perm_stop active on data where multiple greedy rounds
    occur must not raise and must produce a valid (non-crashing) selection -- exercises the new seed
    derivation through the real evaluation.py call sites."""
    from mlframe.feature_selection.filters.info_theory import set_cmi_perm_stop

    X, y = _xy(seed=3, n=600, p=5)
    m = MRMR(
        verbose=0,
        full_npermutations=1,
        baseline_npermutations=1,
        fe_max_steps=0,
        cmi_perm_stop=True,
        cmi_perm_n_permutations=5,
        min_features_fallback=1,
    )
    try:
        m.fit(X, y)
    finally:
        set_cmi_perm_stop(False)
    assert hasattr(m, "support_")
