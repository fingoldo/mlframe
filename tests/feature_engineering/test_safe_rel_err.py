"""Behaviour of the guarded division helpers that replaced additive ``+ eps`` denominators in the audited benchmark diagnostics."""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering._safe_ratio import safe_div, safe_rel_err


def test_safe_div_is_exact_quotient_where_denominator_nonzero() -> None:
    """safe_div returns the exact quotient wherever the denominator is nonzero."""
    num = np.array([1.0, -3.0, 2.5, 1e-300])
    den = np.array([4.0, 0.5, -1.25, 1e-300])
    np.testing.assert_array_equal(safe_div(num, den), num / den)


def test_safe_div_zero_denominator_yields_zero_not_a_pad_artifact() -> None:
    """safe_div returns 0 where the denominator is exactly zero."""
    out = safe_div(np.array([3.0, 0.0, -2.0]), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(out, np.zeros(3))


def test_safe_div_does_not_shrink_a_tiny_denominator() -> None:
    """safe_div does not pad a tiny denominator into a shrunken ratio."""
    # The old ``num / (den + 1e-12)`` returned 1e-12 / 2e-12 = 0.5 here; the exact quotient is 1.0.
    assert safe_div(np.array([2e-12]), np.array([2e-12]))[0] == 1.0


def test_safe_rel_err_matches_plain_ratio_on_nonzero_reference() -> None:
    """safe_rel_err equals the plain ratio where the reference is nonzero."""
    err = np.array([0.5, 1e-3, 2.0])
    ref = np.array([2.0, -4.0, 8.0])
    np.testing.assert_array_equal(safe_rel_err(err, np.abs(ref)), err / np.abs(ref))


def test_safe_rel_err_zero_reference_zero_error_is_zero() -> None:
    """safe_rel_err is 0 when both the error and the reference are zero."""
    assert safe_rel_err(np.array([0.0]), np.array([0.0]))[0] == 0.0


def test_safe_rel_err_zero_reference_nonzero_error_is_inf_not_hidden() -> None:
    """safe_rel_err reports a nonzero error against a zero reference as inf."""
    # A pad would report this mismatch as a finite number (1e-3 / 1e-6 = 1000); the guard must surface it.
    assert np.isinf(safe_rel_err(np.array([1e-3]), np.array([0.0]))[0])


def test_safe_rel_err_propagates_nan_error_even_with_zero_reference() -> None:
    """safe_rel_err propagates a NaN error even against a zero reference."""
    assert np.isnan(safe_rel_err(np.array([np.nan]), np.array([0.0]))[0])


def test_safe_rel_err_scalar_inputs_return_zero_dim_result() -> None:
    """safe_rel_err accepts scalar inputs."""
    assert float(safe_rel_err(0.25, 0.5)) == 0.5
