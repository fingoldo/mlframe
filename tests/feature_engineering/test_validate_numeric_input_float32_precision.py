"""Sensor for validate_numeric_input float32 precision warning.

The feature_engineering/transformer package validates X via validate_numeric_input
then casts to float32 in every downstream operator (kNN distances, SMOTE neighbour
lookups, RFF projections). float32 has a 24-bit mantissa (~16.7M); integer
inputs larger than that lose low bits silently in the cast, corrupting distance
computations without any user-visible error.

Pre-2026-05-20 the validator accepted int inputs unconditionally. The warning
now surfaces the precision loss at validation time so callers can promote to
float64 or rescale upstream.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._utils import validate_numeric_input


def test_safe_int_below_2pow24_no_warning():
    """abs(values) < 2^24 = 16.7M: no precision loss when cast to float32; no warning."""
    X = np.random.default_rng(42).integers(0, 1_000_000, size=(100, 5), dtype=np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_numeric_input(X, name="X")
    assert not any("float32" in str(w.message) for w in caught)


def test_int64_at_2pow24_warns_about_precision():
    """abs-max = 2^24: float32 mantissa exactly cannot represent 2^24 + 1; warn."""
    X = np.full((10, 3), 2**24, dtype=np.int64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_numeric_input(X, name="X")
    f32_warns = [w for w in caught if "float32" in str(w.message)]
    assert len(f32_warns) == 1, f"expected 1 float32-precision warning, got {len(f32_warns)}"
    assert "16,777,216" in str(f32_warns[0].message)


def test_epoch_seconds_in_feature_matrix_warns():
    """A feature column holding epoch-seconds (~1.7e9) routinely sneaks into
    transformer inputs and silently degrades. Validator must warn."""
    rng = np.random.default_rng(42)
    ts = 1_700_000_000 + rng.integers(0, 86_400 * 365, size=1000, dtype=np.int64)
    X = np.column_stack([ts, ts + 100, ts - 50])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_numeric_input(X, name="X")
    f32_warns = [w for w in caught if "float32" in str(w.message)]
    assert len(f32_warns) == 1, "epoch-seconds feature must trigger float32 precision warning"


def test_uint_large_values_also_warn():
    """uint dtype with large values also triggers the warning."""
    X = np.full((10, 3), 2**24 + 1, dtype=np.uint32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_numeric_input(X, name="X")
    f32_warns = [w for w in caught if "float32" in str(w.message)]
    assert len(f32_warns) == 1


def test_float64_input_does_not_warn():
    """float64 caller has already opted into precision; no warning needed."""
    X = np.array([[1.7e9, 2.0e9]] * 10, dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_numeric_input(X, name="X")
    assert not any("float32" in str(w.message) for w in caught)
