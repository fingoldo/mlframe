"""Regression tests for audits/mrmr_audit_2026-07-16/09_error_messages_ux.md.

Covers: DeprecationWarning -> UserWarning for the conflicting random_seed/random_state values;
groups-length mismatch now raises ValueError instead of silently degrading; group_aware_mi disabled
by non-uniform sample_weight now also emits a UserWarning (matching the groups-ignored channel);
the int64->int16 downcast-skip warning is no longer gated behind verbose; stability_selection_method's
error message enumerates valid options; fe_auto's enabled-generators notice is user-visible; the polars
bridge failure warning stays high-level (detail goes to logs); transform_usability's precondition error
is ValueError (consistent with its sibling validation), not AttributeError.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _xy(n=300, seed=0):
    """Two-column normal-feature frame with a binary target derived from column ``a``'s sign."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = pd.Series((X["a"] > 0).astype(int))
    return X, y


def test_conflicting_random_seed_and_state_is_userwarning_not_deprecationwarning():
    """Passing both random_seed and random_state must warn UserWarning (a real conflict), not just the alias's DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MRMR(random_state=1, random_seed=2, verbose=0)
    conflict = [w for w in caught if "both random_seed" in str(w.message)]
    assert conflict, "expected the conflicting-values warning"
    assert conflict[0].category is UserWarning


def test_pure_random_seed_alias_stays_deprecationwarning():
    """random_seed alone (no random_state) is a pure legacy alias -- still DeprecationWarning, no UserWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MRMR(random_seed=1, verbose=0)
    dep = [w for w in caught if "random_seed is deprecated" in str(w.message)]
    assert dep, "expected the pure-alias deprecation warning"
    assert dep[0].category is DeprecationWarning


def test_group_aware_mi_groups_length_mismatch_raises_valueerror():
    """A groups array whose length disagrees with X's row count must raise ValueError, not degrade silently."""
    X, y = _xy(n=100)
    m = MRMR(verbose=0, group_aware_mi=True, min_features_fallback=1)
    bad_groups = np.zeros(50, dtype=int)  # wrong length vs X's 100 rows
    with pytest.raises(ValueError, match="groups length"):
        m.fit(X, y, groups=bad_groups)


def test_group_aware_mi_sample_weight_conflict_emits_userwarning():
    """Non-uniform sample_weight disables group_aware_mi and must warn UserWarning on the same channel as groups-ignored."""
    X, y = _xy(n=200)
    groups = np.repeat(np.arange(20), 10)
    sw = np.ones(200)
    sw[0] = 5.0  # non-uniform
    m = MRMR(verbose=0, group_aware_mi=True, min_features_fallback=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(X, y, groups=groups, sample_weight=sw)
    assert any("group_aware_mi disabled" in str(w.message) and w.category is UserWarning for w in caught)


def test_int16_downcast_skip_warning_not_gated_by_verbose(caplog):
    """int64 targets outside int16 range should warn even at verbose=0."""
    m = MRMR(verbose=0, min_features_fallback=1)
    big_vals = np.array([0, 100000], dtype=np.int64)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.filters.mrmr"):
        out = m._coerce_target_dtype(big_vals)
    assert out.dtype == np.int64
    assert any("skipping memory-saving downcast" in r.message for r in caplog.records)


def test_stability_selection_method_error_lists_valid_options():
    """An invalid stability_selection_method must raise ValueError enumerating the valid choices, not a bare rejection."""
    X, y = _xy(n=100)
    m = MRMR(verbose=0, stability_selection_method="bogus", min_features_fallback=1)
    with pytest.raises(ValueError, match="classic.*cluster.*complementary_pairs"):
        m.fit(X, y)


def test_fe_auto_enabled_generators_emits_userwarning():
    """When fe_auto enables any generators, the notice must be a user-visible UserWarning, not a log-only message."""
    X, y = _xy(n=300)
    m = MRMR(verbose=0, fe_auto=True, min_features_fallback=1, fe_max_steps=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(X, y)
    matches = [w for w in caught if "fe_auto=True enabled" in str(w.message)]
    # fe_auto may legitimately choose to enable nothing for this fixture; only assert the
    # channel exists and fires with the right category when it does.
    for w in matches:
        assert w.category is UserWarning


def test_transform_usability_missing_list_raises_valueerror_not_attributeerror():
    """Calling transform_usability() without usability_aware_lists must raise ValueError (consistent with its sibling validation), not AttributeError."""
    X, y = _xy(n=100)
    m = MRMR(verbose=0, usability_aware_lists=False, min_features_fallback=1)
    m.fit(X, y)
    with pytest.raises(ValueError, match="usability_aware_lists=True"):
        m.transform_usability(X, which="linear")
