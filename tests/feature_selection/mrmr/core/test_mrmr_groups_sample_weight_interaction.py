"""mrmr_audit_2026-07-20 edge_cases.md #187: groups= combined with non-uniform sample_weight must
disable group_aware_mi with a clear, guaranteed-visible warning (not a logger.warning a plain-script
user would never see), rather than silently resampling rows and misaligning them against groups."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=0, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


def _dataset(n=300, seed=0):
    """A trivial classification frame plus a groups= array and a non-uniform sample_weight."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = pd.Series((X["a"] > 0).astype(int))
    groups = rng.integers(0, 6, n)
    sample_weight = rng.uniform(0.1, 5.0, n)
    return X, y, groups, sample_weight


def test_group_aware_mi_disabled_with_warning_under_non_uniform_sample_weight():
    """group_aware_mi=True + groups= + non-uniform sample_weight must fire the documented UserWarning
    naming exactly this reason, and the fit must still complete (group-aware MI silently skipped for
    this fit rather than crashing or silently misaligning rows against groups)."""
    X, y, groups, sample_weight = _dataset()
    m = MRMR(**_kw(group_aware_mi=True, strict_groups=False))
    with pytest.warns(UserWarning, match="group_aware_mi disabled this fit"):
        m.fit(X, y, groups=groups, sample_weight=sample_weight)
    assert hasattr(m, "support_")


def test_group_aware_mi_stays_enabled_with_uniform_sample_weight():
    """The SAME groups= + group_aware_mi=True combination with sample_weight=None (or effectively
    uniform) must NOT fire the disable-warning -- the guard is specific to non-uniform weights."""
    X, y, groups, _sw = _dataset(seed=1)
    m = MRMR(**_kw(group_aware_mi=True, strict_groups=False))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        m.fit(X, y, groups=groups, sample_weight=None)
    assert not any("group_aware_mi disabled this fit" in str(r.message) for r in records)
    assert hasattr(m, "support_")
