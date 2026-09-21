"""Two estimators unpickled from the same bytes must not share their injected default containers.

``__setstate__`` seeds missing attributes from a module-level roster and then mutates the dict it was given. If that roster's mutable values
were handed out by reference, every instance unpickled in a joblib worker would share one list, and appending to one estimator's roster would
silently show up in another's. This is the invariant any narrowing of the copy has to preserve.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr._mrmr_setstate_defaults import (
    _SETSTATE_LEGACY_DEFAULTS,
    build_setstate_defaults,
)


def _mutable_keys():
    """Roster keys whose default is a container, i.e. the ones that could be aliased."""
    return [k for k, v in _SETSTATE_LEGACY_DEFAULTS.items() if isinstance(v, (list, dict, set))]


def test_the_roster_template_is_never_handed_out_by_reference():
    """Every mutable default must be a fresh object, and mutating it must not reach the module-level template."""
    keys = _mutable_keys()
    assert keys, "fixture precondition: the roster is expected to carry mutable defaults"
    first, second = build_setstate_defaults(), build_setstate_defaults()
    for key in keys:
        assert first[key] is not _SETSTATE_LEGACY_DEFAULTS[key], f"{key} aliases the module-level template"
        assert first[key] is not second[key], f"{key} is shared between two builds"
        assert first[key] == _SETSTATE_LEGACY_DEFAULTS[key]
    for key in keys:
        value = first[key]
        if isinstance(value, list):
            value.append("touched")
        elif isinstance(value, set):
            value.add("touched")
        else:
            value["touched"] = True
    assert build_setstate_defaults() == _SETSTATE_LEGACY_DEFAULTS, "mutating one build leaked into the template"


def test_the_build_matches_a_full_deep_copy():
    """Narrowing which values get copied must not change WHAT the roster contains."""
    import copy

    assert build_setstate_defaults() == copy.deepcopy(_SETSTATE_LEGACY_DEFAULTS)


@pytest.mark.parametrize("attr", ["_engineered_features_", "_passthrough_features_"])
def test_two_instances_unpickled_from_the_same_bytes_do_not_share_a_roster_list(attr):
    """The end-to-end form: append to one unpickled estimator's injected list, the other must not see it."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(4)})
    y = (X["c0"] + X["c1"] > 0).astype(np.int64).to_numpy()
    blob = pickle.dumps(MRMR(random_state=0, verbose=0, fe_max_steps=0).fit(X, y))

    first, second = pickle.loads(blob), pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
    if not isinstance(getattr(first, attr, None), list):
        pytest.skip(f"{attr} is not a list on this fitted estimator")
    getattr(first, attr).append("sentinel")
    assert "sentinel" not in getattr(second, attr), f"{attr} is shared between two unpickled instances"
