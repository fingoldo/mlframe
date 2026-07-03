"""MRMR critique batch B: small clear correctness fixes.

- EX-1: _is_argmax_eligible finiteness guard was a tautology (a[isfinite(a)] then isfinite is always True), so NaN
  columns were never excluded from argmax/gate FE.
- S-F5: an unrecognised redundancy_aggregator string silently degraded to plain Fleuret; it now raises.
- ST-3: the empty-screen fallback support_ is int64 (was int32 on Windows).
"""
import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._conditional_gate_fe import _is_argmax_eligible


def test_is_argmax_eligible_excludes_nonfinite_columns():
    assert _is_argmax_eligible(np.array([1.0, 2.0, 3.0])) is True
    assert _is_argmax_eligible(np.arange(5)) is True  # integer finite
    assert _is_argmax_eligible(np.array([1.0, np.nan, 3.0])) is False, "NaN column must be excluded"
    assert _is_argmax_eligible(np.array([1.0, np.inf])) is False, "inf column must be excluded"
    assert _is_argmax_eligible(np.array([], dtype=np.float64)) is True  # empty is vacuously eligible
    assert _is_argmax_eligible(np.array(["a", "b"], dtype=object)) is False  # non-numeric


def test_redundancy_aggregator_typo_raises():
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(0, 4, size=(120, 5)).astype(float), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.random(120) < 0.5).astype(int))
    with pytest.raises(ValueError, match="redundancy_aggregator"):
        MRMR(redundancy_aggregator="JMIM", full_npermutations=1, cv=2,
             run_additional_rfecv_minutes=False).fit(X, y)


def test_valid_redundancy_aggregators_accepted():
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.integers(0, 4, size=(120, 5)).astype(float), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.random(120) < 0.5).astype(int))
    for agg in (None, "jmim", "auto"):
        m = MRMR(redundancy_aggregator=agg, full_npermutations=1, cv=2,
                 run_additional_rfecv_minutes=False)
        m.fit(X, y)  # must not raise
        assert hasattr(m, "support_")
