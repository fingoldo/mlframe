"""biz_value + unit tests for ``calibration.isotonic_risk.isotonic_overfit_risk``.

The win: on a small, noisy calibration set, isotonic regression fits a jagged step function that tracks
individual points' noise (high segment_ratio, correctly FLAGGED) and generalizes poorly to a fresh held-out
set from the same distribution (large in-sample-vs-holdout ECE gap) — exactly the Elo Merchant 7th-place
failure mode ("isotonic overfit both on cv and lb"). On a large, well-behaved calibration set the same
generative process produces a smooth fit (low segment_ratio, NOT flagged) that generalizes properly (small
in-sample-vs-holdout ECE gap). The flag concretely predicts the real generalization gap, not just a
heuristic count.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.calibration.isotonic_risk import isotonic_overfit_risk
from mlframe.calibration.policy import _ece_score


def _make_calibration_data(n: int, noise: float, seed: int):
    """Builds seeded synthetic test data; returns ``(p_raw, y)``."""
    rng = np.random.default_rng(seed)
    p_raw = rng.uniform(0.0, 1.0, size=n)
    true_prob = np.clip(p_raw + noise * rng.standard_normal(n), 0.0, 1.0)
    y = (rng.random(n) < true_prob).astype(np.float64)
    return p_raw, y


def test_isotonic_overfit_risk_flags_small_noisy_fit():
    """Isotonic overfit risk flags small noisy fit."""
    p, y = _make_calibration_data(n=60, noise=0.35, seed=0)
    result = isotonic_overfit_risk(p, y, segment_ratio_threshold=0.05)
    assert result["flagged"] is True
    assert result["n_samples"] == 60


def test_isotonic_overfit_risk_does_not_flag_large_smooth_fit():
    """Isotonic overfit risk does not flag large smooth fit."""
    p, y = _make_calibration_data(n=5000, noise=0.05, seed=1)
    result = isotonic_overfit_risk(p, y, segment_ratio_threshold=0.05)
    assert result["flagged"] is False


def test_isotonic_overfit_risk_returns_reusable_fitted_isotonic():
    """Isotonic overfit risk returns reusable fitted isotonic."""
    p, y = _make_calibration_data(n=200, noise=0.1, seed=2)
    result = isotonic_overfit_risk(p, y)
    preds = result["isotonic_fit"].predict(np.array([0.0, 0.5, 1.0]))
    assert preds.shape == (3,)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_isotonic_overfit_risk_length_mismatch_raises():
    """Isotonic overfit risk length mismatch raises."""
    with pytest.raises(ValueError):
        isotonic_overfit_risk(np.array([0.1, 0.2, 0.3]), np.array([0.0, 1.0]))


def test_isotonic_overfit_risk_too_few_samples_raises():
    """Isotonic overfit risk too few samples raises."""
    with pytest.raises(ValueError):
        isotonic_overfit_risk(np.array([0.5]), np.array([1.0]))


def test_biz_val_flagged_isotonic_fit_generalizes_worse_than_unflagged():
    """Flagged isotonic fit generalizes worse than unflagged."""
    rng_seed = 42
    # SAME noise level, DIFFERENT calibration-set sizes -- the noise/complexity ratio is what should
    # differ, isolating "overfits due to too few samples for the noise level" as the mechanism.
    small_p, small_y = _make_calibration_data(n=50, noise=0.30, seed=rng_seed)
    large_p, large_y = _make_calibration_data(n=4000, noise=0.30, seed=rng_seed + 1)

    small_result = isotonic_overfit_risk(small_p, small_y, segment_ratio_threshold=0.05)
    large_result = isotonic_overfit_risk(large_p, large_y, segment_ratio_threshold=0.05)
    assert small_result["flagged"] is True
    assert large_result["flagged"] is False

    # Fresh held-out set from the SAME generative process (noise=0.30) to measure real generalization.
    holdout_p, holdout_y = _make_calibration_data(n=8000, noise=0.30, seed=rng_seed + 100)

    small_iso = small_result["isotonic_fit"]
    large_iso = large_result["isotonic_fit"]

    ece_small_insample = _ece_score(small_y, small_iso.predict(small_p))
    ece_small_holdout = _ece_score(holdout_y, small_iso.predict(holdout_p))
    ece_large_insample = _ece_score(large_y, large_iso.predict(large_p))
    ece_large_holdout = _ece_score(holdout_y, large_iso.predict(holdout_p))

    small_gap = ece_small_holdout - ece_small_insample
    large_gap = ece_large_holdout - ece_large_insample

    # The flagged (small/noisy) fit should show a materially larger in-sample-to-holdout ECE degradation
    # than the unflagged (large/smooth) fit -- the flag is predicting a REAL generalization gap.
    assert small_gap > large_gap + 0.02, f"flagged fit should generalize measurably worse: small_gap={small_gap:.4f} large_gap={large_gap:.4f}"


def test_biz_val_isotonic_overfit_risk_remediate_beats_plain_isotonic_on_sparse_segment():
    # A small, sparse, noisy calibration set (same shape as the "flags a small noisy fit" fixture above)
    # where the TRUE underlying relationship is a smooth monotonic curve in p (true_prob = p + noise,
    # clipped) -- exactly the shape Platt scaling is built for. Plain isotonic free-form-fits every point's
    # noise (correctly FLAGGED); remediation blends toward Platt everywhere since the whole set is sparse
    # relative to segment_ratio_threshold, recovering the smooth relationship instead of the jagged one.
    """Isotonic overfit risk remediate beats plain isotonic on sparse segment."""
    rng_seed = 42
    calib_p, calib_y = _make_calibration_data(n=50, noise=0.30, seed=rng_seed)

    result = isotonic_overfit_risk(calib_p, calib_y, segment_ratio_threshold=0.05, remediate=True)
    assert result["flagged"] is True
    assert result["remediated"] is True
    assert result["predict"] is not None

    # Fresh held-out set from the SAME generative process.
    holdout_p, holdout_y = _make_calibration_data(n=8000, noise=0.30, seed=rng_seed + 100)

    plain_pred = result["isotonic_fit"].predict(holdout_p)
    remediated_pred = result["predict"](holdout_p)

    ece_plain = _ece_score(holdout_y, plain_pred)
    ece_remediated = _ece_score(holdout_y, remediated_pred)

    # The remediated blend must beat plain isotonic by a real margin, not just tie within noise.
    assert (
        ece_remediated < ece_plain - 0.01
    ), f"remediated ECE should beat plain isotonic by a real margin: ece_plain={ece_plain:.4f} ece_remediated={ece_remediated:.4f}"
