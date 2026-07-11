"""biz_value test for ``calibration.asymmetric_rescale`` (``fit_asymmetric_rescale``, ``apply_asymmetric_rescale``).

The win (8th_ubiquant-market-prediction.md): a model whose predictions are systematically over/under-scaled
DIFFERENTLY for positive vs negative regimes (a genuine asymmetric miscalibration, not overfitting noise)
should have that fixed by a sign-conditional rescale factor. To respect this idea's own known overfitting
risk, the factor is fit on one validation split and evaluated on a SEPARATE HELD-OUT split, confirming a
genuine, non-overfit improvement rather than an in-sample-only artifact.
"""
from __future__ import annotations

import numpy as np

from mlframe.calibration.asymmetric_rescale import apply_asymmetric_rescale, cross_validate_asymmetric_rescale, fit_asymmetric_rescale


def _make_asymmetric_miscalibration_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=n)
    # the model systematically shrinks negative predictions toward zero relative to positive ones (a real,
    # consistent asymmetric miscalibration -- e.g. a loss function that penalizes negative-side errors less).
    y_pred = np.where(y_true < 0, y_true * 0.6, y_true * 1.0) + rng.normal(scale=0.1, size=n)
    return y_true, y_pred


def _neg_mse(y_true, y_pred):
    return -float(np.mean((y_true - y_pred) ** 2))


def test_biz_val_asymmetric_rescale_generalizes_to_held_out_split():
    y_true_val, y_pred_val = _make_asymmetric_miscalibration_dataset(n=1500, seed=0)
    y_true_test, y_pred_test = _make_asymmetric_miscalibration_dataset(n=1500, seed=1)

    fit_result = fit_asymmetric_rescale(y_true_val, y_pred_val, _neg_mse, factor_range=(1.0, 2.0), n_factors=50)

    rescaled_test = apply_asymmetric_rescale(y_pred_test, fit_result["factor"])
    metric_rescaled = _neg_mse(y_true_test, rescaled_test)
    metric_uncorrected = _neg_mse(y_true_test, y_pred_test)

    assert fit_result["factor"] > 1.0, f"expected the fitted factor to genuinely correct the asymmetric shrinkage (factor > 1.0), got {fit_result['factor']:.4f}"
    assert metric_rescaled > metric_uncorrected, f"expected the rescale (fit on a SEPARATE validation split) to generalize to held-out test data, got rescaled={metric_rescaled:.4f} uncorrected={metric_uncorrected:.4f}"


def test_apply_asymmetric_rescale_exact_values():
    y_pred = np.array([-2.0, -1.0, 1.0, 2.0])
    out = apply_asymmetric_rescale(y_pred, factor=1.5)
    np.testing.assert_allclose(out, [-3.0, -1.5, 1.0 / 1.5, 2.0 / 1.5])


def test_apply_asymmetric_rescale_factor_one_is_noop():
    y_pred = np.array([-2.0, -1.0, 1.0, 2.0])
    out = apply_asymmetric_rescale(y_pred, factor=1.0)
    np.testing.assert_allclose(out, y_pred)


def test_biz_val_cross_validate_asymmetric_rescale_flags_stable_vs_unstable_fit():
    # genuine, consistent asymmetric miscalibration at large n: every fold sees the same underlying signal,
    # so the fitted factor should be nearly identical fold-to-fold (low CV, flagged stable).
    y_true_stable, y_pred_stable = _make_asymmetric_miscalibration_dataset(n=4000, seed=42)
    stable_result = cross_validate_asymmetric_rescale(
        y_true_stable, y_pred_stable, _neg_mse, n_folds=5, factor_range=(1.0, 2.0), n_factors=50, seed=0
    )

    # pure noise, no real asymmetric relationship, tiny n, heavy-tailed (Cauchy) so a handful of outliers per
    # fold dominate the grid search: each fold's 1-D search chases whichever outliers landed in its training
    # split, so the fitted factor should swing wildly fold-to-fold (high CV, flagged unstable).
    rng = np.random.default_rng(1)
    n_noise = 25
    y_true_noise = rng.standard_cauchy(size=n_noise) * 0.3
    y_pred_noise = rng.standard_cauchy(size=n_noise) * 0.3
    noise_result = cross_validate_asymmetric_rescale(
        y_true_noise, y_pred_noise, _neg_mse, n_folds=5, factor_range=(1.0, 2.0), n_factors=50, seed=0
    )

    assert stable_result["is_stable"], f"expected the genuine-signal fit to be flagged stable, got factor_cv={stable_result['factor_cv']:.4f}"
    assert not noise_result["is_stable"], f"expected the pure-noise fit to be flagged unstable, got factor_cv={noise_result['factor_cv']:.4f}"
    assert noise_result["factor_cv"] > 3 * stable_result["factor_cv"], (
        f"expected the noise fit's fold-to-fold factor CV to be much larger than the genuine-signal fit's, "
        f"got noise_cv={noise_result['factor_cv']:.4f} stable_cv={stable_result['factor_cv']:.4f}"
    )


def test_cross_validate_asymmetric_rescale_does_not_change_default_fit_behavior():
    # regression guard: the new opt-in CV mode must not alter fit_asymmetric_rescale/apply_asymmetric_rescale
    # when they're called directly (bit-identical to pre-extension behavior).
    y_true_val, y_pred_val = _make_asymmetric_miscalibration_dataset(n=1500, seed=0)
    fit_before = fit_asymmetric_rescale(y_true_val, y_pred_val, _neg_mse, factor_range=(1.0, 2.0), n_factors=50)

    # exercise the new function first, then re-run the original fit to confirm no shared-state contamination.
    cross_validate_asymmetric_rescale(y_true_val, y_pred_val, _neg_mse, n_folds=5, factor_range=(1.0, 2.0), n_factors=50)
    fit_after = fit_asymmetric_rescale(y_true_val, y_pred_val, _neg_mse, factor_range=(1.0, 2.0), n_factors=50)

    assert fit_before == fit_after, f"expected fit_asymmetric_rescale to be bit-identical before/after using cross_validate_asymmetric_rescale, got {fit_before} vs {fit_after}"
