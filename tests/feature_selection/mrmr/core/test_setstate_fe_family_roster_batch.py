"""Per-family engineered-feature rosters that the legacy-pickle roster backfills.

Each FE family seeds its ``<family>_features_`` list in the cascade that owns it, and the roster now gives each one an empty-list default. An
estimator unpickled from an older release then carries the same empty list a fresh fit leaves for a family that did not run, instead of no
attribute at all. Every backfilled list is a fresh object per instance, so no two unpickled estimators share one.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.mrmr._mrmr_setstate_defaults import _SETSTATE_LEGACY_DEFAULTS

_BACKFILLED = (
    "cat_pair_features_",
    "cat_triple_features_",
    "grouped_agg_features_",
    "grouped_delta_features_",
    "grouped_quantile_features_",
    "lagged_diff_features_",
    "missingness_count_features_",
    "missingness_indicator_features_",
    "missingness_pattern_features_",
    "pairwise_log_ratio_features_",
    "pairwise_modular_features_",
    "pairwise_ratio_features_",
)


@pytest.fixture(scope="module")
def fitted() -> MRMR:
    """A small MRMR fit whose fitted state every round-trip test reloads."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(4)})
    y = (X["c0"] + X["c1"] > 0).astype(np.int64).to_numpy()
    return MRMR(random_state=0, verbose=0, fe_max_steps=0).fit(X, y)


@pytest.mark.parametrize("name", _BACKFILLED)
def test_family_roster_value_survives_pickle_round_trip(fitted, name):
    """A pickled and reloaded instance must carry the same family roster it was pickled with.

    The roster is set to a non-empty value first, so the round trip cannot pass by reproducing an empty list.
    """
    probe = MRMR.__new__(MRMR)
    probe.__dict__.update(fitted.__dict__)
    probe.__dict__[name] = ["c0_x_c1"]
    reloaded = pickle.loads(pickle.dumps(probe))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert reloaded.__dict__[name] == ["c0_x_c1"]


@pytest.mark.parametrize("name", _BACKFILLED)
def test_old_pickle_missing_family_roster_is_backfilled_with_empty_list(fitted, name):
    """A pickle without the roster, as an older release wrote it, must load with the empty default rather than no attribute at all."""
    state = fitted.__getstate__()
    state.pop(name, None)
    legacy = MRMR.__new__(MRMR)
    legacy.__setstate__(state)
    assert name in legacy.__dict__, f"{name} is absent after loading a legacy pickle"
    assert legacy.__dict__[name] == []
    assert legacy.__dict__[name] == _SETSTATE_LEGACY_DEFAULTS[name]


def test_backfilled_family_rosters_are_not_shared_across_instances(fitted):
    """Two legacy-loaded instances must not alias one list, otherwise appending to one roster leaks into the other."""
    state = fitted.__getstate__()
    for name in _BACKFILLED:
        state.pop(name, None)
    first = MRMR.__new__(MRMR)
    first.__setstate__(dict(state))
    second = MRMR.__new__(MRMR)
    second.__setstate__(dict(state))
    first.__dict__["cat_pair_features_"].append("leak")
    assert second.__dict__["cat_pair_features_"] == []
