"""Fitted attributes that the legacy-pickle roster backfills, plus the one runtime cache that must never be pickled.

An estimator unpickled from an older release lacks any fitted attribute added since. Each attribute in ``_BACKFILLED`` now has a roster entry,
so the loaded instance carries the default a fresh instance would hold. ``_engineered_names_cache_`` is a memo that ``__getstate__`` drops, so
it must not appear in a pickle and must come back as its empty default.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.mrmr._mrmr_setstate_defaults import _SETSTATE_LEGACY_DEFAULTS

_BACKFILLED = (
    "_fe_budget_fingerprint_",
    "_fe_budget_prev_",
    "_fe_budget_wall_pre_fit_",
    "fe_family_budget_",
    "_fe_recommended_flags_",
    "_identity_cache_ycorr_",
    "_own_last_identity_fp_",
    "_synergy_auto_decision_",
    "degenerate_columns_",
    "groups_ignored_",
)


@pytest.fixture(scope="module")
def fitted() -> MRMR:
    """A small MRMR fit whose fitted state every round-trip test reloads."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(4)})
    y = (X["c0"] + X["c1"] > 0).astype(np.int64).to_numpy()
    return MRMR(random_state=0, verbose=0, fe_max_steps=0).fit(X, y)


_SAMPLE_VALUES = {
    "_fe_budget_fingerprint_": "abc123",
    "_fe_budget_prev_": {"hinge": 0.5},
    "_fe_budget_wall_pre_fit_": {"hinge": 1.25},
    "fe_family_budget_": {"hinge": 0.5},
    "_fe_recommended_flags_": {"hinge_enable": True},
    "_identity_cache_ycorr_": 0.75,
    "_own_last_identity_fp_": "fp-1",
    "_synergy_auto_decision_": {"jmim_engaged": False, "detector_failed": False},
    "degenerate_columns_": {"c9": "constant"},
    "groups_ignored_": True,
}


@pytest.mark.parametrize("name", _BACKFILLED)
def test_fitted_attribute_value_survives_pickle_round_trip(fitted, name):
    """A pickled and reloaded instance must carry the same value for the attribute.

    Attributes that a plain fit with FE disabled never sets are given a representative value first, so the pickle path is still exercised.
    """
    probe = MRMR.__new__(MRMR)
    probe.__dict__.update(fitted.__dict__)
    probe.__dict__.setdefault(name, _SAMPLE_VALUES[name])
    reloaded = pickle.loads(pickle.dumps(probe))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert reloaded.__dict__[name] == probe.__dict__[name]


@pytest.mark.parametrize("name", _BACKFILLED)
def test_old_pickle_missing_attribute_is_backfilled_with_roster_default(fitted, name):
    """A pickle without the attribute, as an older release wrote it, must load with the roster default rather than no attribute at all."""
    state = fitted.__getstate__()
    state.pop(name, None)
    legacy = MRMR.__new__(MRMR)
    legacy.__setstate__(state)
    assert name in legacy.__dict__, f"{name} is absent after loading a legacy pickle"
    assert legacy.__dict__[name] == _SETSTATE_LEGACY_DEFAULTS[name]


def test_engineered_names_cache_is_dropped_from_the_pickle(fitted):
    """The get_feature_names_out memo holds recipe-list identity, so it must be excluded from the pickled state."""
    probe = MRMR.__new__(MRMR)
    probe.__dict__.update(fitted.__dict__)
    probe._engineered_names_cache_ = ([], ["stale"])
    assert "_engineered_names_cache_" not in probe.__getstate__()


def test_engineered_names_cache_reloads_as_empty_default(fitted):
    """After a reload the memo is rebuilt on first use, so it must come back as None rather than a stale tuple."""
    probe = MRMR.__new__(MRMR)
    probe.__dict__.update(fitted.__dict__)
    probe._engineered_names_cache_ = ([], ["stale"])
    reloaded = pickle.loads(pickle.dumps(probe))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert reloaded.__dict__["_engineered_names_cache_"] is None


def test_pre_fit_ctor_snapshot_is_transient_and_gone_after_fit(fitted):
    """The ctor-params snapshot is fit scratch that the fit's restore step pops, so no fitted instance carries it."""
    assert "_pre_fit_ctor_params_snapshot_" not in fitted.__dict__
