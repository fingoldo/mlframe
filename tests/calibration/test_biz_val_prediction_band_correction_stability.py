"""biz_value test for ``calibration.prediction_band_correction.assess_prediction_band_stability``.

Source gap: ``find_prediction_band_shift`` fits its multiplicative correction on whichever OOF rows fall
inside a hard-edged ``(lo, hi]`` band, with no signal about how much evidence backs that fit. A band with
thousands of OOF rows and a genuine subpopulation shift should be trusted; a band with a handful of rows
where the "shift" is mostly resampling noise should not be. This test proves ``assess_prediction_band_stability``
tells the two apart with real numeric thresholds, not just qualitatively.
"""
from __future__ import annotations

import numpy as np

from mlframe.calibration.prediction_band_correction import assess_prediction_band_stability


def test_biz_val_assess_prediction_band_stability_dense_band_is_reliable():
    # plenty of OOF rows in-band, genuine systematic over-prediction (same construction as the
    # existing calibration biz_value test) -- the bootstrap distribution should be tight.
    rng = np.random.default_rng(0)
    n = 4000
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = np.clip(y_true * 0.6 + rng.normal(scale=0.25, size=n) + 0.2, 0, 1)
    mask_high = y_pred > 0.4
    y_pred_biased = y_pred.copy()
    y_pred_biased[mask_high] = np.clip(y_pred_biased[mask_high] * 1.5, 0, 1)

    report = assess_prediction_band_stability(y_true, y_pred_biased, lo=0.4, hi=1.0, n_bootstrap=500, random_state=0)

    assert report.band_n >= 2000, f"expected a dense band, got band_n={report.band_n}"
    assert report.relative_std <= 0.05, f"expected a tight bootstrap spread (relative_std<=0.05), got {report.relative_std:.4f}"
    assert report.is_stable, f"expected a dense genuine shift to be reported stable, got {report}"


def test_biz_val_assess_prediction_band_stability_sparse_band_is_unreliable():
    # a band with only a handful of OOF rows -- the measured "shift" is dominated by which few rows
    # happened to land there rather than a real subpopulation effect; bootstrap resampling should reveal
    # a wide, untrustworthy spread even though find_prediction_band_shift alone reports a confident-looking
    # single float with no warning.
    rng = np.random.default_rng(3)
    n = 4000
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = np.clip(rng.normal(loc=0.15, scale=0.08, size=n), 0, 1)

    report = assess_prediction_band_stability(y_true, y_pred, lo=0.4, hi=1.0, n_bootstrap=500, random_state=0)

    assert report.band_n < 30, f"expected a sparse band (band_n<30), got band_n={report.band_n}"
    assert report.relative_std >= 0.3, f"expected a wide bootstrap spread (relative_std>=0.3) for a noise-driven shift, got {report.relative_std:.4f}"
    assert not report.is_stable, f"expected a sparse noise-driven shift to be reported unreliable, got {report}"
