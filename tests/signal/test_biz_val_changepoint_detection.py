"""biz_value test for ``signal.detect_regime_changepoints``.

The win: on a series with genuine structural breaks (three distinct steady-state levels), the detector should
recover the true breakpoints exactly, and using per-detected-regime baselines instead of one global baseline
should recover close to the true noise floor RMSE instead of being badly blurred by averaging across regimes.
Also verifies the effect-size filter suppresses a spurious tiny-magnitude break.
"""
from __future__ import annotations

import numpy as np

from mlframe.signal.changepoint_detection import detect_regime_changepoints


def test_biz_val_detect_regime_changepoints_recovers_true_breaks_and_improves_baseline_rmse():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100), rng.normal(3, 1, 100)])

    result = detect_regime_changepoints(y, min_segment_length=15, penalty=5.0, min_effect_size=0.5)
    assert result["breakpoints"] == [100, 200]
    assert result["n_regimes"] == 3

    global_pred = np.full_like(y, y.mean())
    global_rmse = float(np.sqrt(np.mean((y - global_pred) ** 2)))

    regime_pred = np.zeros_like(y)
    for r in range(result["n_regimes"]):
        mask = result["regime_id"] == r
        regime_pred[mask] = y[mask].mean()
    regime_rmse = float(np.sqrt(np.mean((y - regime_pred) ** 2)))

    assert regime_rmse < global_rmse * 0.4, (
        f"regime-aware baselines should recover close to the true noise floor vs a badly-blurred global baseline: "
        f"regime={regime_rmse:.4f} global={global_rmse:.4f}"
    )
    assert regime_rmse < 1.3  # close to the true noise std of 1.0


def test_detect_regime_changepoints_filters_negligible_effect_size():
    rng = np.random.default_rng(1)
    # a tiny, practically-negligible mean shift (0.1) buried in noise (std=1) should NOT be flagged.
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(0.1, 1, 100)])
    result = detect_regime_changepoints(y, min_segment_length=15, penalty=10.0, min_effect_size=0.5)
    assert result["n_regimes"] == 1
    assert result["breakpoints"] == []


def test_detect_regime_changepoints_short_series_returns_single_regime():
    y = np.array([1.0, 2.0, 3.0])
    result = detect_regime_changepoints(y, min_segment_length=10)
    assert result["n_regimes"] == 1
    assert (result["regime_id"] == 0).all()
