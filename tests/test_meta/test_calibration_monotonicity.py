"""C4 — meta-test that post-hoc isotonic calibration **never makes the
Brier score worse** on its own training data, by construction.

The Murphy-1973 decomposition guarantees that an isotonic-regression
calibrator minimises the reliability term on the data it was fitted to,
without changing the resolution or uncertainty terms. Therefore::

    Brier(isotonic(p), y) ≤ Brier(p, y)   on the calibration set

This must hold per-label for every miscalibrated input. The only failure
modes are:
  * a bug in ``_PerClassIsotonicCalibrator`` (e.g. swapping a sign,
    fitting on a degenerate single-class fold and silently emitting
    identity, but with the wrong sign-convention);
  * a regression in ``fast_brier_score_loss`` (e.g. accidentally
    computing log-loss instead);
  * an off-by-one in the per-label dispatch.

Hypothesis-driven over the (n, K, miscalibration-magnitude) cube.
``_PerClassIsotonicCalibrator`` is multilabel/multiclass only, so the
test exercises the multilabel path — the multiclass case has the same
math via the per-class binary decomposition.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st


@settings(deadline=None, max_examples=15,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=200, max_value=800),
    k=st.integers(min_value=2, max_value=4),
    miscal=st.floats(min_value=-0.4, max_value=0.4,
                     allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_isotonic_calibration_does_not_worsen_per_label_brier(n, k, miscal, seed):
    """Per-label Brier on the calibrator's training set must not
    increase after isotonic. Tests the multilabel path (1 calibrator
    per label, fit + applied on the same data)."""
    from mlframe.training.configs import TargetTypes
    from mlframe.training.trainer import _PerClassIsotonicCalibrator
    from mlframe.metrics.core import fast_brier_score_loss

    rng = np.random.default_rng(seed)
    p_true = rng.random((n, k))
    y = (rng.random((n, k)) < p_true).astype(np.int8)
    # Reject pathological cases: constant labels make the calibrator a no-op.
    for col in range(k):
        assume(0 < int(y[:, col].sum()) < n)

    p_miscal = np.clip(p_true + miscal, 1e-3, 1 - 1e-3)

    # method="isotonic" is REQUIRED here: the Murphy-1973 "never worsens training Brier" guarantee is
    # exact only for the free-form monotone isotonic fit. The calibrator now DEFAULTS to method="sigmoid"
    # (a 2-parameter Platt fit that generalises better on held-out slices but, being parametric, can
    # raise training Brier slightly on already-calibrated inputs), so this isotonic property test must
    # request isotonic explicitly rather than rely on the default.
    cal = _PerClassIsotonicCalibrator.fit(
        p_miscal, y, TargetTypes.MULTILABEL_CLASSIFICATION, method="isotonic",
    )
    calibrated = cal.predict_proba(p_miscal)

    for col in range(k):
        before = fast_brier_score_loss(
            y[:, col].astype(np.float64), p_miscal[:, col],
        )
        after = fast_brier_score_loss(
            y[:, col].astype(np.float64), calibrated[:, col],
        )
        assert after <= before + 1e-6, (
            f"isotonic calibration worsened per-label Brier on training "
            f"data (col={col}): before={before:.6f}, after={after:.6f}, "
            f"delta=+{after - before:.2e} (n={n}, k={k}, miscal={miscal:.3f}, "
            f"seed={seed})"
        )


def test_isotonic_calibration_constant_label_returns_identity():
    """When a per-label fold has only one class (constant y), the
    calibrator should fall back to identity for that column. Catches
    a future regression that crashes on degenerate folds.
    """
    from mlframe.training.configs import TargetTypes
    from mlframe.training.trainer import _PerClassIsotonicCalibrator

    rng = np.random.default_rng(42)
    n, k = 200, 3
    p = rng.random((n, k))
    y = np.zeros((n, k), dtype=np.int8)  # constant: all-negative across all labels

    cal = _PerClassIsotonicCalibrator.fit(
        p, y, TargetTypes.MULTILABEL_CLASSIFICATION,
    )
    out = cal.predict_proba(p)
    assert out.shape == p.shape
    # Identity expected when all calibrators are None (constant labels).
    np.testing.assert_allclose(out, p, atol=1e-9)


def test_brier_score_in_unit_interval_for_random_input():
    """Sanity for ``fast_brier_score_loss`` itself — Brier on
    [0, 1] inputs must stay in [0, 1]. Backstop for a metric refactor."""
    from mlframe.metrics.core import fast_brier_score_loss
    rng = np.random.default_rng(42)
    for _ in range(5):
        n = rng.integers(50, 500)
        p = rng.random(n)
        y = (rng.random(n) < p).astype(np.float64)
        score = fast_brier_score_loss(y, p)
        assert 0.0 <= score <= 1.0, f"Brier {score} outside [0, 1]"
