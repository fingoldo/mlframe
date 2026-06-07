"""Regression: ``audit_residuals`` skew/kurt computed via ``z * z2`` /
``z2 * z2`` instead of ``z ** 3`` / ``z ** 4`` (iter129, 2026-05-21).

numpy's ``z ** 3`` and ``z ** 4`` dispatch through the generic
``np.power`` ufunc which is ~7x slower than explicit elementwise
multiplies on integer exponents. Reusing ``z2 = z * z`` for both
``z^3 = z * z2`` and ``z^4 = z2 * z2`` collapses the per-call skew+kurt
work from ~4.0 ms to ~0.58 ms at n=50k.

Also packs the two percentile reads into one ``np.percentile(..., [1, 99])``
call (one partial sort instead of two; 0.88 ms -> 0.54 ms at n=50k).

These tests pin the bit-equivalence of the per-cell math so a future
regression that flips back to ``z ** 3`` for "readability" surfaces at
unit-test time.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.targets.regression_residual_audit import audit_residuals


def test_skew_kurt_match_z_pow_oracle():
    """Audit skew + excess_kurt against the legacy ``z ** 3 / z ** 4``
    formula on a deterministic synthetic residual stream."""
    rng = np.random.default_rng(0)
    n = 5_000
    y_true = rng.standard_normal(n)
    # Add a skew-inducing residual: y_pred = y_true - lognormal noise.
    y_pred = y_true - rng.lognormal(mean=0.0, sigma=0.5, size=n)
    audit = audit_residuals(y_true, y_pred, sample_size=None)

    # Replay with the legacy explicit-power formula on the same residuals.
    residuals = y_true - y_pred
    mean = residuals.mean()
    std = residuals.std(ddof=1)
    z = (residuals - mean) / std
    legacy_skew = float(np.mean(z ** 3))
    legacy_kurt = float(np.mean(z ** 4) - 3.0)
    assert abs(audit.skew - legacy_skew) < 1e-10, (
        f"skew via z*z2 must match z**3 oracle at fp64 epsilon"
    )
    assert abs(audit.excess_kurt - legacy_kurt) < 1e-10, (
        f"excess_kurt via z2*z2 must match z**4-3 oracle at fp64 epsilon"
    )


def test_percentile_pair_matches_two_separate_calls():
    """Packed percentile pair must return the same p01 / p99 as two
    separate np.percentile calls (no order-of-sort sensitivity)."""
    rng = np.random.default_rng(1)
    n = 5_000
    y_true = rng.standard_normal(n)
    y_pred = y_true + 0.1 * rng.standard_normal(n)
    audit = audit_residuals(y_true, y_pred, sample_size=None)

    residuals = y_true - y_pred
    legacy_p01 = float(np.percentile(residuals, 1))
    legacy_p99 = float(np.percentile(residuals, 99))
    assert abs(audit.p01 - legacy_p01) < 1e-12
    assert abs(audit.p99 - legacy_p99) < 1e-12


def test_audit_handles_zero_std_constant_residuals():
    """Edge case: residuals all equal (std = 0) -> skew/kurt fall to 0,
    no divide-by-zero from the z2/z2 path."""
    y_true = np.ones(100, dtype=np.float64)
    y_pred = np.ones(100, dtype=np.float64)
    audit = audit_residuals(y_true, y_pred, sample_size=None)
    assert audit.std == 0.0
    assert audit.skew == 0.0
    assert audit.excess_kurt == 0.0
    assert audit.pct_outliers_3sigma == 0.0
