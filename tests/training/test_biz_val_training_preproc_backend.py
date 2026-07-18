"""biz_value coverage for decision-influencing PreprocessingBackendConfig knobs.

Covers the fields that change the produced transform / engineered set:
  - ``robust_q_low`` / ``robust_q_high`` -- the robust-scaler quantile spread,
    which bounds outlier magnitude in the scaled output (wide default spread
    divides by a large IQR and shrinks outlier rows; a tight IQR explodes them).
  - ``polynomial_max_features`` -- the auto-tune cap that flips interaction_only,
    then decrements degree, then skips, so the engineered column count stays
    under the cap.

Quantitative floors are set well below the measured deltas (deterministic, no
seed variation in the magnitudes asserted) so a regression that silently
ignores either knob trips the test.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlframe.training.configs import PreprocessingBackendConfig
from mlframe.training.feature_handling.polynomial import PolynomialFeatureExpander
from mlframe.training.pipeline import create_polarsds_pipeline


def _robust_scaled_max_abs(x: np.ndarray, q_low: float, q_high: float) -> float:
    """Applies the robust-scaler preprocessing backend and returns the max absolute scaled value."""
    df = pl.DataFrame({"a": x.astype("float32")})
    cfg = PreprocessingBackendConfig(
        scaler_name="robust",
        imputer_strategy=None,
        categorical_encoding=None,
        robust_q_low=q_low,
        robust_q_high=q_high,
    )
    pipe = create_polarsds_pipeline(df, cfg, verbose=0)
    if pipe is None:
        pytest.skip("polars-ds unavailable")
    out = pipe.transform(df).to_numpy()
    return float(np.nanmax(np.abs(out)))


def test_biz_val_backend_robust_quantile_spread_bounds_outliers():
    """Wide robust quantiles (0.01/0.99, the default) divide by a large IQR that
    spans the outliers, so the scaled magnitude of a heavy-outlier feature stays
    O(1); a tight IQR (0.25/0.75) excludes the outliers from the spread and the
    same rows explode by orders of magnitude. The knob therefore directly
    decides how much a few outlier rows dominate a downstream linear model.

    Measured: tight max|z|~743 vs wide max|z|~3.6 -> ratio ~200x. Floor 20x.
    """
    rng = np.random.RandomState(0)
    n = 800
    x = rng.randn(n)
    idx = rng.choice(n, 20, replace=False)
    x[idx] += rng.randn(20) * 500.0

    tight = _robust_scaled_max_abs(x, 0.25, 0.75)
    wide = _robust_scaled_max_abs(x, 0.01, 0.99)

    assert wide < 10.0, f"wide-quantile robust scale should bound outliers, got max|z|={wide:.2f}"
    assert (
        tight / wide >= 20.0
    ), f"robust_q_low/high must change outlier bounding: tight={tight:.2f} wide={wide:.2f} ratio={tight / wide:.1f} (<20x means the quantiles were ignored)"


def test_biz_val_backend_polynomial_max_features_caps_engineered_set():
    """``polynomial_max_features`` auto-tunes the degree-3 full expansion down
    until the projected column count fits the cap (flip interaction_only ->
    decrement degree -> skip). With 20 inputs the uncapped degree-3 set is 1770
    columns; cap=200 must auto-tune to a strictly smaller set under the cap.

    Measured: uncapped 1770 cols (deg=3) vs cap=200 -> 20 cols (deg=1, io=True).
    """
    X = np.random.RandomState(0).randn(50, 20).astype("float32")

    uncapped = PolynomialFeatureExpander(degree=3, interaction_only=False, max_features_out=None).fit(X)
    cols_unc = np.asarray(uncapped.transform(X)).shape[1]

    capped = PolynomialFeatureExpander(degree=3, interaction_only=False, max_features_out=200).fit(X)
    cols_cap = np.asarray(capped.transform(X)).shape[1]

    assert cols_unc > 1000, f"degree-3 on 20 inputs should explode; got {cols_unc} cols"
    assert cols_cap <= 200, f"cap=200 must hold: engineered set has {cols_cap} cols"
    assert cols_cap < cols_unc, "cap must shrink the engineered set vs uncapped"
    assert capped.effective_degree < uncapped.effective_degree or bool(
        capped.effective_interaction_only
    ), f"cap should auto-tune degree down or flip interaction_only; eff_degree={capped.effective_degree} eff_io={capped.effective_interaction_only}"
