"""TRAINING_COMPOSITE_ENSEMBLE_ESTIMATOR_TRANSFORMS-2 regression test: linear_residual_multi_robust must
be covered by ADDITIVE_BASE_TRANSFORMS (and thus is_enabled), matching its byte-identical forward/inverse
sibling linear_residual_multi.

The bug (fixed): linear_residual_multi_robust reuses the exact same forward/inverse functions as
linear_residual_multi (registry: "Forward / inverse identical to linear_residual_multi once (alphas, beta)
are fitted") but was missing from ADDITIVE_BASE_TRANSFORMS, so it silently never got the soft-base-shrink
OOD guard its byte-identical sibling gets.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlframe.training.composite.estimator._soft_shrink import ADDITIVE_BASE_TRANSFORMS, BASE_FIT_RANGE_KEY, is_enabled
from mlframe.training.composite.transforms import get_transform

pytestmark = pytest.mark.fast


def test_linear_residual_multi_robust_in_additive_base_transforms():
    """linear_residual_multi_robust must be in ADDITIVE_BASE_TRANSFORMS, matching its sibling."""
    assert "linear_residual_multi" in ADDITIVE_BASE_TRANSFORMS
    assert "linear_residual_multi_robust" in ADDITIVE_BASE_TRANSFORMS


def test_is_enabled_true_for_linear_residual_multi_robust_with_fit_range():
    """is_enabled must return True for linear_residual_multi_robust once a fit-range is captured, the
    same as it already does for linear_residual_multi."""
    transform = get_transform("linear_residual_multi_robust")
    assert transform.requires_base
    self_stub = SimpleNamespace(soft_base_shrink=True)
    params = {BASE_FIT_RANGE_KEY: {"lo": 0.0, "hi": 1.0, "iqr": 0.5}}
    assert is_enabled(self_stub, transform, params) is True


def test_is_enabled_matches_sibling_linear_residual_multi():
    """Both siblings must agree on is_enabled given the same fit-range params."""
    self_stub = SimpleNamespace(soft_base_shrink=True)
    params = {BASE_FIT_RANGE_KEY: {"lo": 0.0, "hi": 1.0, "iqr": 0.5}}
    t_multi = get_transform("linear_residual_multi")
    t_robust = get_transform("linear_residual_multi_robust")
    assert is_enabled(self_stub, t_multi, params) == is_enabled(self_stub, t_robust, params) is True


def test_second_diff_in_additive_base_transforms():
    """TRAINING_COMPOSITE_ENSEMBLE_ESTIMATOR_TRANSFORMS-3: second_diff's inverse
    (y = T_hat + 2*b1 - b2) is purely additive-linear in base, same family as diff
    (which IS covered), so it must also be in ADDITIVE_BASE_TRANSFORMS."""
    assert "diff" in ADDITIVE_BASE_TRANSFORMS
    assert "second_diff" in ADDITIVE_BASE_TRANSFORMS


def test_is_enabled_true_for_second_diff_with_fit_range():
    """is_enabled must return True for second_diff once a fit-range is captured, matching diff."""
    self_stub = SimpleNamespace(soft_base_shrink=True)
    params = {BASE_FIT_RANGE_KEY: {"lo": 0.0, "hi": 1.0, "iqr": 0.5}}
    t_diff = get_transform("diff")
    t_second_diff = get_transform("second_diff")
    assert is_enabled(self_stub, t_diff, params) == is_enabled(self_stub, t_second_diff, params) is True
