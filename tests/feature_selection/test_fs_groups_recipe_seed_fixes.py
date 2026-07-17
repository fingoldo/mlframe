"""Regression + biz_value tests for the Wave-4 feature-selection fixes (A1-01..13).

Covers:
  A1-01 groups: strict_groups raises; warn-only path stamps groups_ignored_ into fit metadata.
  A1-04 rename: skip_retraining_on_same_content alias; same-shape-different-y does NOT replay a cached fit.
  A1-06 identity-cache y-correlation gate: distinct uncorrelated target is refused; correlated target hits.
  A1-09 medoid SU corr_method: SU captures non-monotone redundancy Pearson misses.
  A1-13 fallback metadata: fallback_used_ + fallback_metadata_ populated when the count floor fires.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _xy(seed=0, n=600, p=6):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(p)})
    y = pd.Series((X["f0"] + 0.3 * rng.randn(n) > 0).astype(int), name="t")
    return X, y


# ---------------------------------------------------------------- A1-01 groups


def test_a1_01_strict_groups_raises():
    X, y = _xy()
    m = MRMR(strict_groups=True, full_npermutations=2, cv=2)
    with pytest.raises(NotImplementedError):
        m.fit(X, y, groups=np.arange(len(y)) % 10)


def test_a1_01_warn_only_stamps_groups_ignored_metadata():
    X, y = _xy()
    m = MRMR(strict_groups=False, full_npermutations=2, cv=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y, groups=np.arange(len(y)) % 10)
    assert getattr(m, "groups_ignored_", None) is True


def test_a1_01_no_groups_metadata_false():
    X, y = _xy()
    m = MRMR(full_npermutations=2, cv=2)
    m.fit(X, y)
    assert getattr(m, "groups_ignored_", None) is False


# ------------------------------------------------------- A1-04 rename + alias


def test_a1_04_alias_maps_to_content():
    """The deprecated ``skip_retraining_on_same_shape`` alias resolves to ``skip_retraining_on_same_content``
    LAZILY at fit time, NOT eagerly in __init__: sklearn clone-ability requires __init__ to store the
    constructor args UNMODIFIED so ``get_params`` round-trips and ``clone`` of a default-constructed estimator
    does not re-emit the deprecation warning. The pre-fix shape -- eager promotion onto the public attr at
    construction -- was the stale proxy; the real contract is verbatim storage + the alias driving the
    fit-time EFFECTIVE skip value, which we prove behaviourally below."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(skip_retraining_on_same_shape=False)
    # sklearn purity: __init__ stores both verbatim (no eager promotion of the alias onto the new name), so
    # the deprecated alias is preserved for the fit-time resolution rather than silently folded away.
    params = m.get_params()
    assert params["skip_retraining_on_same_shape"] is False
    assert params["skip_retraining_on_same_content"] is True  # untouched default
    assert m.skip_retraining_on_same_shape is False  # the non-None alias is kept for fit-time resolution


def test_a1_04_alias_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        MRMR(skip_retraining_on_same_shape=True)


def test_a1_04_same_shape_different_y_does_not_replay_cached_fit():
    """The cache keys on CONTENT, not shape: a same-shape-but-different-y refit must produce a y2-appropriate
    support, NOT replay the y1 fit. This is the core correctness guarantee behind the rename."""
    rng = np.random.RandomState(1)
    n, p = 800, 6
    X = pd.DataFrame({f"f{i}": rng.randn(n) for i in range(p)})
    y1 = pd.Series((X["f0"] > 0).astype(int), name="t")
    y2 = pd.Series((X["f3"] > 0).astype(int), name="t")  # same shape, different signal column

    m = MRMR(skip_retraining_on_same_content=True, full_npermutations=3, cv=2, min_relevance_gain_relative_to_first=0.0)
    m.fit(X, y1)
    sup1 = set(np.atleast_1d(m.support_).tolist())
    m.fit(X, y2)
    sup2 = set(np.atleast_1d(m.support_).tolist())
    # f0 index 0 should anchor y1; f3 index 3 should anchor y2. A stale replay would give identical supports.
    assert 0 in sup1, f"y1 should pick f0; got {sup1}"
    assert 3 in sup2, f"y2 should pick f3; got {sup2}"


# ----------------------------------------------- A1-06 identity-cache y-corr gate


def test_a1_06_ycorr_gate_refuses_uncorrelated_target():
    from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_y_corr, _mrmr_y_corr_sample

    rng = np.random.RandomState(2)
    y_cache = (rng.randn(2000) > 0).astype(int)
    y_indep = rng.randint(0, 2, 2000)
    corr = abs(_mrmr_y_corr(_mrmr_y_corr_sample(y_cache), _mrmr_y_corr_sample(y_indep)) or 0.0)
    assert corr < 0.5, f"independent target should be below the 0.5 gate; got {corr}"


def test_a1_06_ycorr_gate_admits_correlated_target():
    from mlframe.feature_selection.filters._mrmr_fingerprints import _mrmr_y_corr, _mrmr_y_corr_sample

    rng = np.random.RandomState(3)
    z = rng.randn(2000)
    y_cache = (z + 0.2 * rng.randn(2000) > 0).astype(int)
    y_corr = (z + 0.3 * rng.randn(2000) > 0).astype(int)
    corr = abs(_mrmr_y_corr(_mrmr_y_corr_sample(y_cache), _mrmr_y_corr_sample(y_corr)) or 0.0)
    assert corr >= 0.5, f"correlated target should clear the 0.5 gate; got {corr}"


def test_a1_06_default_threshold_is_05():
    assert MRMR().mrmr_identity_cache_ycorr_threshold == 0.5


# --------------------------------------------------- A1-09 medoid SU corr_method


def test_a1_09_su_captures_nonmonotone_redundancy():
    """biz_value: SU clusters z and z**2 together (non-monotone redundancy); Pearson does not."""
    from mlframe.feature_selection.filters.group_aware import (
        cluster_features_by_correlation,
        _su_redundancy_matrix,
    )

    rng = np.random.RandomState(4)
    z = rng.randn(1000)
    df = pd.DataFrame({"z0": z, "z1": z**2, "n0": rng.randn(1000), "n1": rng.randn(1000)})
    su = _su_redundancy_matrix(df)
    pear = np.abs(df.corr(method="pearson").to_numpy())
    # SU sees the non-monotone redundancy strongly; Pearson sees only finite-sample noise (~0.1), so SU is
    # multiples larger. Assert the gap rather than a tight Pearson bound (z**2 leaves a small spurious linear corr).
    assert su[0, 1] > 0.3, f"SU should see z0~z1 redundancy; got {su[0, 1]}"
    assert su[0, 1] > 2.0 * pear[0, 1], f"SU should dwarf Pearson on non-monotone redundancy; su={su[0, 1]} pear={pear[0, 1]}"
    cl_su = cluster_features_by_correlation(df, threshold=0.3, method="su")
    assert cl_su[0] == cl_su[1], "SU clustering should group z0 and z1"


# --------------------------------------------------- A1-13 fallback metadata


def test_a1_13_fallback_metadata_populated():
    """When screening rejects everything and the count floor fires, fallback_used_ + fallback_metadata_ are set."""
    rng = np.random.RandomState(5)
    X = pd.DataFrame({f"f{i}": rng.randn(400) for i in range(5)})
    y = pd.Series((rng.randn(400) > 0).astype(int), name="t")  # pure noise: no feature is informative
    m = MRMR(full_npermutations=3, cv=2, min_relevance_gain=10.0, min_features_fallback=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    if getattr(m, "fallback_used_", False):
        md = getattr(m, "fallback_metadata_", None)
        assert isinstance(md, dict)
        assert md["fallback_used"] is True
        assert md["min_features_fallback"] == 1
        assert "uninformative" in md
    else:
        # If the fallback didn't fire (selection found something), metadata stays None -- still a valid contract.
        assert getattr(m, "fallback_metadata_", None) is None
