"""Audit 2026-11-20 finding 35-F21: the OOF cache key tells apart calls that differ in sample weights or component specs.

The key used to leave both out, so a caller reusing a cache_key with other weights got the old matrix back.
"""

import numpy as np

from mlframe.training.composite.ensemble import _oof_full_cache_key

_BASE = dict(cache_key="k", kfold=3, random_state=0, holdout_frac=0.2, time_ordering=None, external_holdout_X=None, external_holdout_y=None, group_ids=None)


def test_sample_weight_changes_the_key():
    """Equal weights give equal keys; different weights give different keys."""
    w1 = np.ones(10)
    w2 = np.ones(10)
    w2[3] = 2.0
    assert _oof_full_cache_key(**_BASE, sample_weight=w1) == _oof_full_cache_key(**_BASE, sample_weight=w1.copy())
    assert _oof_full_cache_key(**_BASE, sample_weight=w1) != _oof_full_cache_key(**_BASE, sample_weight=w2)


def test_component_specs_change_the_key_including_mid_array_edits():
    """A spec array differing only in the middle (which repr would elide) changes the key."""
    a = np.arange(5000, dtype=float)
    b = a.copy()
    b[2500] = -1.0
    s1 = [{"name": "c", "fitted_params": {"alpha": a}}]
    s2 = [{"name": "c", "fitted_params": {"alpha": b}}]
    assert _oof_full_cache_key(**_BASE, component_specs=s1) != _oof_full_cache_key(**_BASE, component_specs=s2)
    assert _oof_full_cache_key(**_BASE, component_specs=s1) == _oof_full_cache_key(
        **_BASE, component_specs=[{"name": "c", "fitted_params": {"alpha": a.copy()}}]
    )
