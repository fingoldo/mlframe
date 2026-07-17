"""Regression pin for code-quality audit finding #19: MRMR inherits sklearn's ``SelectorMixin``.

Confirms the MRO surgery landed correctly: ``SelectorMixin`` is a real base (isinstance holds, and
``sklearn``'s mask-derived helpers like ``inverse_transform`` become available), while MRMR's own
``get_feature_names_out``/``get_support``/``transform`` still win over ``SelectorMixin``'s mask-only
versions -- MRMR's ``transform`` can add FE-engineered columns, so ``SelectorMixin.transform``'s
mask-only slice would be the wrong contract if it won.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectorMixin

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.mrmr._mrmr_class_transform import _MRMRTransformMixin


def _fit_small():
    """Fit a tiny, fast MRMR instance for MRO/contract assertions."""
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(int)
    m = MRMR(full_npermutations=1, baseline_npermutations=1, verbose=0, fe_max_steps=0)
    m.fit(X, y)
    return X, m


def test_mrmr_is_selectormixin_instance():
    """MRMR must be a real SelectorMixin subclass (finding #19), not just duck-typed."""
    assert issubclass(MRMR, SelectorMixin)
    _, m = _fit_small()
    assert isinstance(m, SelectorMixin)


def test_mrmr_get_feature_names_out_is_own_mixin_not_selectormixin():
    """_MRMRTransformMixin's get_feature_names_out must win over SelectorMixin's via MRO ordering."""
    assert MRMR.get_feature_names_out is _MRMRTransformMixin.get_feature_names_out


def test_mrmr_get_support_is_own_mixin_not_selectormixin():
    """_MRMRTransformMixin's get_support must win over SelectorMixin's via MRO ordering."""
    assert MRMR.get_support is _MRMRTransformMixin.get_support


def test_mrmr_transform_still_returns_engineered_columns():
    """MRMR's own transform() (class-body-defined, beats any inherited SelectorMixin.transform
    regardless of MRO) must still be the one that runs -- a mask-only SelectorMixin.transform would
    silently drop FE-engineered columns."""
    X, m = _fit_small()
    out = m.transform(X)
    assert out.shape[0] == X.shape[0]
    assert len(m.get_feature_names_out()) == (out.shape[1] if hasattr(out, "shape") else len(out.columns))


def test_mrmr_selectormixin_inverse_transform_available():
    """SelectorMixin's inverse_transform (mask-derived) becomes usable once _get_support_mask is wired."""
    X, m = _fit_small()
    mask = m._get_support_mask()
    assert mask.dtype == bool
    assert mask.shape[0] == X.shape[1]
    assert mask.sum() == len(m.support_)
