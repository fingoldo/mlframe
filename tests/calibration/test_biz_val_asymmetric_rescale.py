"""biz_value test for ``calibration.asymmetric_rescale`` (``fit_asymmetric_rescale``, ``apply_asymmetric_rescale``).

The win (8th_ubiquant-market-prediction.md): a model whose predictions are systematically over/under-scaled
DIFFERENTLY for positive vs negative regimes (a genuine asymmetric miscalibration, not overfitting noise)
should have that fixed by a sign-conditional rescale factor. To respect this idea's own known overfitting
risk, the factor is fit on one validation split and evaluated on a SEPARATE HELD-OUT split, confirming a
genuine, non-overfit improvement rather than an in-sample-only artifact.
"""
from __future__ import annotations

import numpy as np

from mlframe.calibration.asymmetric_rescale import apply_asymmetric_rescale, fit_asymmetric_rescale


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
