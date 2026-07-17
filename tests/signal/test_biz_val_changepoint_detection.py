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


def test_detect_regime_changepoints_njit_backend_matches_ruptures_backend():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100), rng.normal(3, 1, 100)])

    njit_result = detect_regime_changepoints(y, min_segment_length=15, penalty=5.0, min_effect_size=0.5, backend="njit")
    ruptures_result = detect_regime_changepoints(y, min_segment_length=15, penalty=5.0, min_effect_size=0.5, backend="ruptures")

    assert njit_result["breakpoints"] == ruptures_result["breakpoints"]
    assert njit_result["n_regimes"] == ruptures_result["n_regimes"]
    assert (njit_result["regime_id"] == ruptures_result["regime_id"]).all()


def test_detect_regime_changepoints_unknown_backend_raises():
    y = np.arange(100, dtype=np.float64)
    try:
        detect_regime_changepoints(y, min_segment_length=10, backend="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_detect_regime_changepoints_segment_stats_opt_in_default_unchanged():
    # regression: return_segment_stats defaults to False and must not alter any prior key, bit-identical.
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100), rng.normal(3, 1, 100)])

    baseline = detect_regime_changepoints(y, min_segment_length=15, penalty=5.0, min_effect_size=0.5)
    with_stats = detect_regime_changepoints(y, min_segment_length=15, penalty=5.0, min_effect_size=0.5, return_segment_stats=True)

    assert "segment_stats" not in baseline
    assert baseline["breakpoints"] == with_stats["breakpoints"]
    assert baseline["n_regimes"] == with_stats["n_regimes"]
    assert (baseline["regime_id"] == with_stats["regime_id"]).all()
    with_stats.pop("segment_stats")
    assert baseline.keys() == with_stats.keys()


def test_detect_regime_changepoints_segment_stats_short_series():
    y = np.array([1.0, 2.0, 3.0])
    result = detect_regime_changepoints(y, min_segment_length=10, return_segment_stats=True)
    assert result["n_regimes"] == 1
    assert len(result["segment_stats"]) == 1
    stat = result["segment_stats"][0]
    assert stat["start"] == 0 and stat["end"] == 3 and stat["count"] == 3
    assert stat["mean"] == y.mean()


def test_biz_val_detect_regime_changepoints_segment_stats_match_manual_slice_and_feed_regime_features():
    """The win: return_segment_stats gives exact per-regime mean/std/count without the caller re-slicing y,
    and using those regime-level stats as a per-row feature set predicts y far better than a single global
    mean/std feature -- proving the stats are both exactly correct and materially useful."""
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100), rng.normal(3, 1, 100)])

    result = detect_regime_changepoints(y, min_segment_length=15, penalty=5.0, min_effect_size=0.5, return_segment_stats=True)
    assert result["n_regimes"] == 3
    stats = result["segment_stats"]
    assert len(stats) == 3

    # exact correctness vs a manual slice-and-compute using the returned breakpoints.
    bounds = [0] + result["breakpoints"] + [len(y)]
    for i, stat in enumerate(stats):
        manual_segment = y[bounds[i] : bounds[i + 1]]
        assert stat["start"] == bounds[i]
        assert stat["end"] == bounds[i + 1]
        assert stat["count"] == manual_segment.shape[0]
        assert stat["mean"] == float(np.mean(manual_segment))
        assert stat["std"] == float(np.std(manual_segment, ddof=1))

    # material usefulness: per-row regime mean (built directly from segment_stats, no re-slicing) as a
    # 1-feature predictor of y crushes the single global-mean predictor, same comparison as the RMSE test above.
    regime_mean_feature = np.zeros_like(y)
    for r, stat in enumerate(stats):
        regime_mean_feature[result["regime_id"] == r] = stat["mean"]
    regime_rmse = float(np.sqrt(np.mean((y - regime_mean_feature) ** 2)))

    global_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))

    assert regime_rmse < global_rmse * 0.4
    assert regime_rmse < 1.3  # close to the true noise std of 1.0
