"""D1 regression (2026-06-22): engineered-name -> SOURCE recovery must not mis-stem one-hot sources.

The univariate / pair-cross orthogonal FE names engineered columns ``"{src}__{basis_code}{degree}"``.
The pre-fix code recovered the source by ``name.split("__", 1)[0]`` -- which MISPARSES any raw input
whose own name carries ``"__"`` (one-hot / dummy names like ``"city__NY"``): ``"city__NY__He2"`` was
stemmed to ``"city"``, the per-source raw-MI baseline lookup MISSED, the uplift denominator collapsed
to the ``1e-12`` floor, and EVERY such engineered column scored a spurious near-infinite uplift.

These tests pin the corrected behaviour:
* the source is recovered against the KNOWN raw-column set (longest ``"{raw}__"`` prefix), so a
  one-hot source ``"city__NY"`` is recovered intact;
* the uplift denominator therefore matches the column's own raw-MI baseline (NOT the 1e-12 floor);
* normal (no-``"__"``) inputs are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
    _source_from_engineered_name,
    score_features_by_mi_uplift,
)
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_pair_cross_fe import (
    _pair_sources_from_engineered_name,
)


def test_source_recovery_onehot_and_plain():
    raw = ["city__NY", "city__LA", "age", "city"]
    # one-hot source kept intact, not stemmed to "city"
    assert _source_from_engineered_name("city__NY__He2", raw) == "city__NY"
    assert _source_from_engineered_name("city__LA__T3", raw) == "city__LA"
    # plain source unchanged
    assert _source_from_engineered_name("age__He3", raw) == "age"
    # longest-prefix wins: "city" must not shadow "city__NY"
    assert _source_from_engineered_name("city__NY__He2", ["city", "city__NY"]) == "city__NY"
    # a plain "city" engineered column still resolves to "city"
    assert _source_from_engineered_name("city__He2", ["city"]) == "city"


def test_source_recovery_fallback_no_raw_set():
    # No raw name prefixes -> legacy first-"__" split fallback.
    assert _source_from_engineered_name("foo__He2", []) == "foo"
    assert _source_from_engineered_name("foo", []) == "foo"


def test_pair_source_recovery_onehot():
    raw = ["city__NY", "age", "income"]
    assert _pair_sources_from_engineered_name("city__NY*age__He2_He3", raw) == ("city__NY", "age")
    assert _pair_sources_from_engineered_name("age*city__NY__He2_He3", raw) == ("age", "city__NY")
    assert _pair_sources_from_engineered_name("age*income__He2_He3", raw) == ("age", "income")
    # non-pair name -> (None, None)
    assert _pair_sources_from_engineered_name("age__He2", raw) == (None, None)


def test_uplift_denominator_not_floored_for_onehot_source():
    """The crux of the D1 bug: a one-hot source's uplift denominator must be its own
    raw-MI baseline, NOT the 1e-12 floor (which would manufacture a near-infinite uplift)."""
    rng = np.random.default_rng(0)
    n = 2000
    # a genuinely informative one-hot-named source; the engineered column is monotone in it,
    # so engineered MI ~ source MI -> the CORRECT uplift is O(1), the BUGGY uplift is ~ emi/1e-12.
    x = rng.normal(size=n)
    y = (x > 0).astype(np.int64)
    raw_X = pd.DataFrame({"city__NY": x})
    eng_X = pd.DataFrame({"city__NY__He2": x ** 2, "city__NY__He3": x ** 3})

    df = score_features_by_mi_uplift(raw_X, eng_X, y, nbins=10)
    # Every engineered row must attribute to the TRUE one-hot source, not the stem "city".
    assert set(df["source_col"]) == {"city__NY"}
    # Baseline must be the real raw MI (> 0), never the missing-lookup 0.0 -> 1e-12-floor case.
    assert (df["baseline_mi"] > 0).all()
    # With a real positive baseline the uplift stays bounded and modest (not ~1e11).
    assert (df["uplift"] < 1e6).all()


def test_normal_input_unchanged():
    """No-``"__"`` source names: behaviour identical to the legacy stem."""
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.normal(size=n)
    y = (x > 0).astype(np.int64)
    raw_X = pd.DataFrame({"feat": x})
    eng_X = pd.DataFrame({"feat__He2": x ** 2})
    df = score_features_by_mi_uplift(raw_X, eng_X, y, nbins=10)
    assert set(df["source_col"]) == {"feat"}
    assert (df["baseline_mi"] > 0).all()
