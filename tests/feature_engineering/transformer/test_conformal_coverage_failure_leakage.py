"""Leakage regression test for ``compute_conformal_coverage_failure_features``.

The bug (fixed): coverage labels for the h1 half used IN-SAMPLE predictions (``predict_proba``/``predict`` on h1 after the model was fit including h1), and the kNN
bank was built over the FULL fold-train (h1 + h2). The conformal coverage indicator for an h1 row was therefore computed from a memorised in-sample prediction, which
systematically understates miscoverage, and those over-optimistic h1 labels polluted the neighbour aggregates.

The fix restricts both the coverage labels and the kNN bank to the held-out calib half h2, whose predictions are genuinely out-of-sample (model fit on h1 only).

Sensor: with the in-sample h1 labels, ``frac_covered`` for query rows is inflated toward 1.0 because the in-sample residuals are near zero (model memorised h1).
Post-fix the bank holds only honest out-of-sample h2 residuals, so the mean coverage matches the nominal ``1 - alpha`` far better. We pin that the realised mean
coverage is not pathologically inflated above the nominal level the way the in-sample-contaminated bank produced.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import compute_conformal_coverage_failure_features


pytestmark = pytest.mark.fast


def test_conformal_coverage_bank_is_held_out_only():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    n, d = 300, 5
    X = rng.standard_normal((n, d)).astype(np.float32)
    # Hard noisy regression target: a model fit on h1 will MEMORISE h1 (in-sample residual ~0) but mispredict out-of-sample, so an in-sample bank inflates coverage.
    y = (X[:, 0] + 2.0 * rng.standard_normal(n)).astype(np.float32)
    splitter = KFold(n_splits=4, shuffle=True, random_state=42)
    alpha = 0.2

    df = compute_conformal_coverage_failure_features(
        X,
        y,
        None,
        splitter,
        seed=0,
        task="regression",
        k_neighbors=15,
        alpha=alpha,
        standardize=True,
    )
    frac = df["ccf_frac_covered"].to_numpy()
    mean_cov = float(np.nanmean(frac))

    # With an honest held-out-only bank the realised mean neighbour-coverage tracks the nominal 1-alpha=0.8 (measured 0.790 post-fix). The in-sample-contaminated
    # h1 labels (memorised residuals near 0 -> "covered") inflate it to 0.852 pre-fix. Threshold 0.82 sits cleanly between the two regimes.
    assert mean_cov <= 0.82, (
        f"Mean neighbour frac_covered={mean_cov:.3f} is inflated above the nominal 1-alpha=0.8; the coverage bank must use only the held-out calib half h2 "
        "(out-of-sample), not in-sample h1 predictions that memorise residuals to ~0 and over-report coverage."
    )
