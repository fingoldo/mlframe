"""Regression tests for audits/mrmr_audit_2026-07-16/08_sklearn_joblib_compat.md.

Finding #1: n_jobs=-1 / parallel_kwargs=None used to be resolved to concrete values BEFORE
store_params_in_object, so get_params()/clone()/pickle never round-tripped the sentinel.
Finding #2: no pickle schema-version stamp -- downgrade (newer pickle, older code) was undetectable.
Finding #3: __setstate__'s type(self)() fresh-instance construction used to bake the unpickling
worker's own psutil.cpu_count() into a legacy pickle's restored n_jobs.
Finding #4: __repr__'s textual patch of BaseEstimator.__repr__ had no defensive fallback.
"""
from __future__ import annotations

import pickle
import warnings

import pytest
from sklearn.base import clone

from mlframe.feature_selection.filters.mrmr._mrmr_class import MRMR, _MRMR_SCHEMA_VERSION


def test_n_jobs_sentinel_survives_get_params_and_clone():
    """finding #1: n_jobs=-1 is stored raw (not resolved to a core count) so get_params()/clone()
    round-trip the sentinel, not a machine-specific resolved value."""
    m = MRMR(n_jobs=-1, verbose=0)
    assert m.n_jobs == -1
    assert m.get_params()["n_jobs"] == -1
    c = clone(m)
    assert c.n_jobs == -1


def test_parallel_kwargs_sentinel_survives_get_params_and_clone():
    """finding #1: parallel_kwargs=None stays None (not resolved to a concrete dict)."""
    m = MRMR(verbose=0)
    assert m.parallel_kwargs is None
    assert m.get_params()["parallel_kwargs"] is None
    c = clone(m)
    assert c.parallel_kwargs is None


def test_effective_n_jobs_resolves_sentinel():
    """finding #1: _effective_n_jobs() resolves -1 to a positive physical core count at the point of
    use, and passes a concrete positive value through unchanged."""
    m = MRMR(n_jobs=-1, verbose=0)
    assert m._effective_n_jobs() >= 1
    m2 = MRMR(n_jobs=3, verbose=0)
    assert m2._effective_n_jobs() == 3


def test_effective_parallel_kwargs_resolves_sentinel():
    """finding #1: _effective_parallel_kwargs() resolves None to the threading-backend default dict,
    and passes a caller-supplied dict through unchanged (as a copy, not aliased)."""
    m = MRMR(verbose=0)
    resolved = m._effective_parallel_kwargs()
    assert resolved["backend"] == "threading"

    custom = {"backend": "loky", "n_jobs": 2}
    m2 = MRMR(parallel_kwargs=custom, verbose=0)
    resolved2 = m2._effective_parallel_kwargs()
    assert resolved2 == custom
    assert resolved2 is not custom  # not aliased


def test_pickle_stamps_schema_version():
    """finding #2: __getstate__ stamps _mrmr_schema_version; a round-tripped pickle carries it."""
    m = MRMR(verbose=0)
    state = m.__getstate__()
    assert state["_mrmr_schema_version"] == _MRMR_SCHEMA_VERSION
    m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert getattr(m2, "_mrmr_schema_version", None) == _MRMR_SCHEMA_VERSION


def test_setstate_warns_on_newer_schema_version_downgrade():
    """finding #2: unpickling a state stamped with a NEWER schema version than this installed mlframe
    warns (downgrade scenario), rather than silently applying unrecognized state."""
    m = MRMR(verbose=0)
    state = m.__getstate__()
    state["_mrmr_schema_version"] = _MRMR_SCHEMA_VERSION + 1
    m2 = MRMR()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m2.__setstate__(state)
    assert any("schema version" in str(w.message) for w in caught)


def test_setstate_no_warning_on_same_or_missing_schema_version():
    """finding #2: the normal case (same version, or a pre-finding-#2 pickle with no stamp at all)
    produces no downgrade warning."""
    m = MRMR(verbose=0)
    state = m.__getstate__()
    m2 = MRMR()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m2.__setstate__(state)
    assert not any("schema version" in str(w.message) for w in caught)

    state_no_stamp = dict(state)
    state_no_stamp.pop("_mrmr_schema_version", None)
    m3 = MRMR()
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        m3.__setstate__(state_no_stamp)
    assert not any("schema version" in str(w.message) for w in caught2)


def test_legacy_pickle_n_jobs_no_longer_leaks_worker_core_count(monkeypatch):
    """finding #3: a legacy pickle (state missing the n_jobs key entirely) restores n_jobs to the RAW
    ctor default (-1), not a worker-local psutil.cpu_count() value -- n_jobs/parallel_kwargs are no
    longer resolved at construction time, so the fresh-instance catch-all in __setstate__ can no longer
    bake a resolved value into legacy state."""
    m = MRMR(verbose=0)
    state = m.__getstate__()
    state.pop("n_jobs", None)
    m2 = MRMR()
    m2.__setstate__(state)
    assert m2.n_jobs == -1


def test_repr_includes_n_workers_and_never_raises():
    """finding #4: __repr__ still appends n_workers=, and the defensive wrapper never raises even if
    monkeypatched to simulate an unexpected BaseEstimator.__repr__ output shape."""
    m = MRMR(verbose=0)
    r = repr(m)
    assert "n_workers=" in r

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sklearn.base.BaseEstimator.__repr__", lambda self, N_CHAR_MAX=700: "not-a-normal-repr-format")
        r2 = repr(m)  # must not raise even though the trailing-")" assumption is violated
        assert isinstance(r2, str)
