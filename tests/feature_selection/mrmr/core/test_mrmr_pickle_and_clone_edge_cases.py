"""Pickling/clone edge cases (mrmr_audit_2026-07-20 edge_cases.md #175, #179, #223): pickling an
estimator interrupted mid-fit, and clone() independently re-deriving random_state=-1's fresh seed
rather than carrying over the already-resolved concrete seed."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mlframe.feature_selection.filters import MRMR


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=0, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


def _dataset(n=200, seed=0):
    """A trivial classification frame."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = pd.Series((X["a"] > 0).astype(int))
    return X, y


class TestPicklingPostFit:
    """Pins the existing, already-working post-fit pickle round-trip contract."""

    def test_post_fit_pickle_round_trip_preserves_support_and_transform(self):
        """A fully-fitted estimator must unpickle to an identical support_ and transform() output."""
        X, y = _dataset()
        m = MRMR(**_kw(random_seed=42))
        m.fit(X, y)
        restored = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(getattr(restored, "support_", [])) == list(getattr(m, "support_", []))
        pd.testing.assert_frame_equal(restored.transform(X), m.transform(X))


class TestPicklingMidFit:
    """The actual gap: pickling an instance whose fit raised partway through must not crash
    __getstate__ on missing fitted attributes it assumes exist."""

    def test_pickle_after_interrupted_fit_does_not_crash_getstate(self, monkeypatch):
        """Patch an internal fit-body step to raise after construction but before support_ is set;
        catch the exception, then assert pickling the partially-fitted instance either succeeds
        cleanly or raises a clear error -- never a cryptic AttributeError from __getstate__."""
        X, y = _dataset(seed=1)
        m = MRMR(**_kw(random_seed=7))

        def _boom(self, *args, **kwargs):
            """Simulates a fit-body step failing partway through, after construction."""
            raise RuntimeError("simulated mid-fit failure")

        # Patch the CLASS (not the instance) so the interrupted instance's __dict__ never holds an
        # unpicklable local closure -- an instance-level patch would itself break the pickle round-trip
        # this test is trying to isolate, masking the real __getstate__/__setstate__ behavior under test.
        monkeypatch.setattr(type(m), "_fit_impl", _boom, raising=False)
        with pytest.raises(RuntimeError, match="simulated mid-fit failure"):
            m.fit(X, y)
        assert not hasattr(m, "support_")

        try:
            restored = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        except Exception as exc:
            pytest.fail(f"pickling a partially-fitted (never-completed) instance raised an unclear error: {exc!r}")
        assert not hasattr(restored, "support_")


class TestCloneRandomStateMinusOneResolution:
    """random_state=-1 is the documented 'derive a fresh seed' sentinel: clone() must independently
    re-resolve it, never carry over the original's already-resolved concrete seed."""

    def test_clone_keeps_random_state_as_minus_one_not_the_resolved_seed(self):
        """get_params() on the clone must still show random_state=-1, not a concrete derived value."""
        m = MRMR(**_kw(random_state=-1))
        _ = m._effective_random_seed()  # resolve -1 into a concrete seed on the original instance
        cloned = clone(m)
        assert cloned.get_params()["random_state"] == -1
        assert m.get_params()["random_state"] == -1
