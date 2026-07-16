"""Regression test: synthesized placeholder feature names must be ``f{i}`` everywhere.

``_fit_identity_shortcut`` (ndarray fit, cache-hit path) synthesizes names as ``f0``, ``f1``, ...
``_fit_multioutput`` (ndarray fit, 2D y union/intersect path) used to synthesize ``feature_0``,
``feature_1``, ... instead -- an inconsistent convention within the same class. Both paths must
agree, and the ``get_feature_names_out`` back-compat regex fallback (used when the
``_feature_names_in_synthesized_`` sentinel is missing, e.g. an older pickle) must still recognize
BOTH the current ``f\\d+`` convention and the legacy ``feature_\\d+`` convention it replaces.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import MRMR


def _fast(**kw):
    """Build a fast-fitting MRMR instance for these tests, overridable via kwargs."""
    base = dict(full_npermutations=5, baseline_npermutations=3, n_jobs=1, verbose=0, fe_fast_search=False, interactions_max_order=1, random_seed=9)
    base.update(kw)
    return MRMR(**base)


def _xy_2d_y(seed: int = 5, n: int = 160):
    """Build a small synthetic 2D-y fixture (two binary targets driven by columns 0 and 2)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    Y2 = np.column_stack([(X[:, 0] > 0).astype(int), (X[:, 2] > 0).astype(int)])
    return X, Y2


def test_fit_multioutput_ndarray_synthesizes_f_prefixed_names():
    """_fit_multioutput on a bare ndarray must use the ``f{i}`` convention, matching
    _fit_identity_shortcut, not the inconsistent ``feature_{i}`` convention."""
    X, Y2 = _xy_2d_y()
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy="union").fit(X, Y2)
    assert m.multioutput_strategy_ == "union"
    names = [str(n) for n in m.feature_names_in_]
    assert names == [f"f{i}" for i in range(X.shape[1])]
    assert not any(n.startswith("feature_") for n in names)


def test_get_feature_names_out_regex_fallback_recognizes_f_prefixed_names():
    """When the ``_feature_names_in_synthesized_`` sentinel is missing (e.g. an older pickle),
    the regex fallback in get_feature_names_out must still classify ``f{i}``-named features as
    synthesized, so a caller passing new names via ``input_features=`` is honored instead of
    raising a spurious column-drift ValueError."""
    X, Y2 = _xy_2d_y()
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy="union").fit(X, Y2)
    # _fit_multioutput never sets the sentinel, so get_feature_names_out always falls back
    # to the regex heuristic on a multioutput-fitted ndarray input -- exactly the path this
    # test exercises.
    assert not hasattr(m, "_feature_names_in_synthesized_")

    new_names = [f"custom_{i}" for i in range(X.shape[1])]
    out = m.get_feature_names_out(input_features=new_names)
    raw_out = [n for n in out if "(" not in str(n)]
    assert set(raw_out).issubset(set(new_names))


def test_get_feature_names_out_regex_fallback_still_recognizes_legacy_feature_prefix():
    """Back-compat: a genuinely old pickle synthesized under the pre-fix ``feature_{i}``
    convention (sentinel absent) must still be recognized as synthesized."""
    X, Y2 = _xy_2d_y()
    MRMR._FIT_CACHE.clear()
    m = _fast(multioutput_strategy="union").fit(X, Y2)
    legacy_names = [f"feature_{i}" for i in range(X.shape[1])]
    m.feature_names_in_ = np.asarray(legacy_names, dtype=object)
    assert not hasattr(m, "_feature_names_in_synthesized_")

    new_names = [f"custom_{i}" for i in range(X.shape[1])]
    out = m.get_feature_names_out(input_features=new_names)
    raw_out = [n for n in out if "(" not in str(n)]
    assert set(raw_out).issubset(set(new_names))
