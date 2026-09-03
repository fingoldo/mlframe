"""METRICS-4 (2026-08-05 audit): ``integral_calibration_error_from_metrics`` accepted a
``calibration_coverage`` argument but never referenced it in the function body -- every caller's
ICE was bit-identical regardless of coverage. Fixed by adding a ``coverage_weight`` parameter
(default 0.0, so all existing callers are unaffected) wiring in ``(1 - calibration_coverage) *
coverage_weight`` as a loss term. The same dead-parameter bug existed in the batched njit twin
``_batch_per_class_ice_kernel`` (used by ``compute_probabilistic_multiclass_error``'s fastpath),
which didn't even compute a coverage value; fixed in lockstep.
"""

from __future__ import annotations

import numpy as np

from mlframe.metrics.core import (
    integral_calibration_error_from_metrics,
    compute_probabilistic_multiclass_error,
    _batch_per_class_ice_kernel,
)


def test_default_coverage_weight_ignores_calibration_coverage():
    """Pre-fix behaviour: with the default coverage_weight=0.0, ICE must stay bit-identical
    regardless of calibration_coverage's value -- proves the default preserves every existing
    caller's exact prior output."""
    kwargs = dict(calibration_mae=0.1, calibration_std=0.05, brier_loss=0.2, roc_auc=0.7, pr_auc=0.3)
    vals = [integral_calibration_error_from_metrics(calibration_coverage=cov, **kwargs) for cov in (0.0, 0.1, 0.5, 0.9, 1.0)]
    assert len(set(vals)) == 1, f"default coverage_weight=0.0 must ignore calibration_coverage, got {vals}"


def test_nonzero_coverage_weight_penalizes_low_coverage():
    """With an explicit nonzero coverage_weight, lower calibration_coverage must increase ICE
    (this is the mechanism the dead parameter was supposed to provide)."""
    kwargs = dict(calibration_mae=0.1, calibration_std=0.05, brier_loss=0.2, roc_auc=0.7, pr_auc=0.3, coverage_weight=1.0)
    ice_full_cov = integral_calibration_error_from_metrics(calibration_coverage=1.0, **kwargs)
    ice_low_cov = integral_calibration_error_from_metrics(calibration_coverage=0.1, **kwargs)
    assert ice_low_cov > ice_full_cov, (ice_low_cov, ice_full_cov)
    expected_diff = (1.0 - 0.1) - (1.0 - 1.0)
    assert abs((ice_low_cov - ice_full_cov) - expected_diff) < 1e-9


def test_batch_kernel_default_coverage_weight_matches_scalar_default():
    """The batched njit kernel's coverage_weight default (0.0) must reproduce the scalar function's
    ICE for a fully-populated-bins fixture (n_nonempty == nbins), where coverage=1.0 regardless."""
    rng = np.random.default_rng(0)
    n, nbins = 500, 10
    y_true_nk = (rng.random((n, 1)) < 0.4).astype(np.int8)
    y_pred_nk = np.clip(rng.random((n, 1)), 1e-3, 1 - 1e-3)
    desc_idx_nk = np.ascontiguousarray(np.argsort(-y_pred_nk, axis=0).astype(np.int64))

    ice_default = _batch_per_class_ice_kernel(
        y_true_nk, y_pred_nk, desc_idx_nk, nbins, True, 3.0, 2.0, 0.8, 1.5, 0.1, 0.54, 0.0,
    )
    ice_explicit_zero = _batch_per_class_ice_kernel(
        y_true_nk, y_pred_nk, desc_idx_nk, nbins, True, 3.0, 2.0, 0.8, 1.5, 0.1, 0.54, 0.0, 0.0,
    )
    ice_nonzero = _batch_per_class_ice_kernel(
        y_true_nk, y_pred_nk, desc_idx_nk, nbins, True, 3.0, 2.0, 0.8, 1.5, 0.1, 0.54, 0.0, 5.0,
    )
    assert np.allclose(ice_default, ice_explicit_zero)
    # Ties out only if bins happen to be fully populated at this n/nbins; otherwise coverage<1 and terms differ.
    assert not np.allclose(ice_default, ice_nonzero) or ice_nonzero[0] >= ice_default[0]


def test_compute_probabilistic_multiclass_error_coverage_weight_propagates():
    """coverage_weight must reach compute_probabilistic_multiclass_error's fastpath (default 0.0
    bit-identical to omitting it; nonzero changes the result) -- the fastpath uses the batched
    njit kernel directly, the exact call site METRICS-4 originally missed."""
    # Small n with default nbins=10 leaves several bins empty (partial coverage), so a nonzero
    # coverage_weight actually changes the result -- a larger, densely-binned n would make
    # n_nonempty == nbins regardless (coverage == 1.0), masking the propagation this test pins.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 15)
    p = rng.random((15, 3))
    p /= p.sum(axis=1, keepdims=True)

    e_omitted = compute_probabilistic_multiclass_error(y, p)
    e_explicit_zero = compute_probabilistic_multiclass_error(y, p, coverage_weight=0.0)
    e_nonzero = compute_probabilistic_multiclass_error(y, p, coverage_weight=0.5)

    assert e_omitted == e_explicit_zero
    assert e_omitted != e_nonzero
