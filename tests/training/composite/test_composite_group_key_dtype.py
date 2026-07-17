"""Regression: group/category label keys must survive int<->float dtype drift.

The grouped (``linear_residual_grouped``) and high-cardinality
(``target_encoding_residual``) transforms key their per-group dicts by a string
form of the label. A bare ``str`` made the integer ``1`` (``'1'``) and the float
``1.0`` (``'1.0'``) DIFFERENT keys, so fitting on integer category labels then
predicting on the SAME categories arriving as float (a routine polars int->float
promotion / pandas join upcast) missed every key and silently routed every row to
the global fallback -- the learned per-group residual was added back with the wrong
(global) level, producing systematically wrong y with no error or warning.

These tests FAIL on the pre-fix ``str(label)`` keying (every level collapses to the
global mean / global alpha-beta) and PASS once the canonical key collapses
integral-valued int and float labels to the same key.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.transforms import _canonical_group_key
from mlframe.training.composite.transforms.categorical import (
    _category_encoding_lookup,
    _target_encoding_residual_fit,
)
from mlframe.training.composite.transforms.linear import _linear_residual_grouped_fit
from mlframe.training.composite.transforms.nonlinear import _row_alpha_beta


def test_canonical_group_key_collapses_integral_dtypes() -> None:
    """``1``, ``1.0``, np.int64(1), np.float64(1.0) all map to the same key."""
    assert _canonical_group_key(1) == _canonical_group_key(1.0) == _canonical_group_key(np.int64(1)) == _canonical_group_key(np.float64(1.0)) == "1"
    # Non-integral float keeps full precision; string passes through.
    assert _canonical_group_key(2.5) == repr(2.5)
    assert _canonical_group_key("well_A") == "well_A"
    # Bool is not silently treated as 0/1 integer.
    assert _canonical_group_key(True) == "True"


def test_target_encoding_lookup_robust_to_int_float_dtype_shift() -> None:
    """Fit on int category labels, predict on the SAME categories as float.

    Pre-fix: the float lookup misses every ``str(1.0)='1.0'`` key (fit stored
    ``'1'``) and returns the global mean for every row (only ONE distinct encoded
    value). Post-fix: the per-category means are recovered (5 distinct levels).
    """
    rng = np.random.default_rng(1)
    n = 800
    cats_int = rng.integers(0, 5, n)
    y = cats_int * 10.0 + rng.standard_normal(n)

    params = _target_encoding_residual_fit(
        y,
        base=np.zeros(n),
        groups=cats_int,
        smoothing=2.0,
    )
    enc_int = _category_encoding_lookup(cats_int, params)
    enc_float = _category_encoding_lookup(cats_int.astype(np.float64), params)

    # The int and float lookups must agree exactly (same categories).
    assert np.allclose(enc_int, enc_float)
    # And they must NOT all collapse to the global mean (the pre-fix failure).
    assert len(np.unique(np.round(enc_float, 6))) == 5
    assert not np.allclose(enc_float, params["global_mean"])


def test_grouped_linear_residual_predict_robust_to_int_float_dtype_shift() -> None:
    """Fit per-group OLS on int groups, predict on the SAME groups as float.

    Pre-fix: ``_row_alpha_beta`` looks up ``str(1.0)`` against ``'1'`` keys, misses,
    and returns the GLOBAL alpha/beta for every row (one distinct alpha). Post-fix:
    the per-group alpha/beta are recovered (>1 distinct alpha).
    """
    rng = np.random.default_rng(2)
    n = 900
    groups_int = rng.integers(0, 5, n)
    base = rng.standard_normal(n)
    # Each group has its own slope so per-group alphas genuinely differ.
    slopes = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    y = slopes[groups_int] * base + groups_int * 4.0 + rng.standard_normal(n) * 0.1

    params = _linear_residual_grouped_fit(
        y,
        base,
        groups=groups_int,
        min_group_size=10,
    )
    a_int, b_int = _row_alpha_beta(groups_int, params)
    a_float, b_float = _row_alpha_beta(groups_int.astype(np.float64), params)

    assert np.allclose(a_int, a_float)
    assert np.allclose(b_int, b_float)
    # Per-group slopes must be recovered, not collapsed to the single global alpha.
    assert len(np.unique(np.round(a_float, 6))) > 1
