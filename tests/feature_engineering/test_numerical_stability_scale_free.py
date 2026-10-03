"""Scale and offset robustness of variance, ratio and binning kernels in feature engineering.

Each test feeds an input where an additive epsilon pad or a raw power-sum variance used to dominate the true quantity (large offset with a small spread, or a
tiny overall scale) and asserts the stable result; a companion check pins agreement with the plain reference on ordinary data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering._safe_ratio import safe_div
from mlframe.feature_engineering._welford_njit import welford_push, welford_std
from mlframe.feature_engineering.anchor import anchor_residual_rmse_features
from mlframe.feature_engineering.bayesian import kalman_filter_posterior_1d
from mlframe.feature_engineering.ensemble_features import predictor_consensus_entropy
from mlframe.feature_engineering.entity_inter_event import _expanding_stats_njit
from mlframe.feature_engineering.hurst import dfa_alpha, dfa_alpha2_quadratic, higuchi_fd
from mlframe.feature_engineering.spectral import rolling_spectral_centroid
from mlframe.feature_engineering.transformer.class_mahalanobis import _shrunk_covariance
from mlframe.feature_selection.filters._composite_group_agg_fe import generate_composite_group_agg_features
from mlframe.feature_selection.filters._orthogonal_univariate_fe._imbalance_mi import _class_balanced_mi_batch_njit
from mlframe.feature_selection.filters._temporal_agg_fe import _expanding_stat_past_only
from mlframe.feature_selection.filters._temporal_agg_fe_rolling import _rolling_stat_past_only
from mlframe.feature_selection.filters.hermite_fe import _plugin_mi_classif_batch_njit, _quantile_bin_njit, _quantile_bin_numpy


def _epoch_like(n: int = 1000, seed: int = 0) -> np.ndarray:
    """Epoch-second-scale values (1.7e9) with unit spread: the regime where sum-of-squares variance cancels."""
    return 1.7e9 + np.random.default_rng(seed).normal(0.0, 1.0, n)


def test_welford_helper_matches_numpy_std_on_large_offset():
    """The shared Welford primitives reproduce numpy's two-pass std at a 1.7e9 offset where raw power sums return garbage."""
    x = _epoch_like()
    mean, m2 = 0.0, 0.0
    for i, v in enumerate(x):
        mean, m2 = welford_push(i, mean, m2, float(v))
    assert abs(welford_std(m2, x.size, 1) - x.std(ddof=1)) < 1e-6
    assert welford_std(0.0, 1, 1) == 0.0


def test_expanding_std_large_offset_matches_reference():
    """Expanding past-only std on epoch-scale values must track the two-pass std (old raw-sumsq form gave ~51 instead of ~0.98)."""
    x = _epoch_like()
    out = _expanding_stat_past_only(x, np.zeros(x.size, dtype=np.int64), "std")
    assert abs(out[-1] - x[:-1].std(ddof=1)) < 1e-5
    assert abs(out[500] - x[:500].std(ddof=1)) < 1e-5


def test_expanding_std_ordinary_data_matches_reference():
    """On unit-scale data the Welford expanding std agrees with the reference to ~1e-12 relative."""
    y = np.random.default_rng(1).normal(5.0, 2.0, 400)
    out = _expanding_stat_past_only(y, np.zeros(y.size, dtype=np.int64), "std")
    ref = np.array([y[:i].std(ddof=1) for i in range(2, 400)])
    assert np.allclose(out[2:], ref, rtol=1e-12, atol=0.0)
    assert out[1] == 0.0 and np.isnan(out[0])


def test_rolling_std_large_offset_matches_reference():
    """Rolling time-window std on epoch-scale values must match the two-pass std of the in-window past values."""
    x = _epoch_like()
    t = np.arange(x.size, dtype=np.float64)
    out = _rolling_stat_past_only(t, x, np.zeros(x.size, dtype=np.int64), "2000", "std")
    assert abs(out[-1] - x[:-1].std(ddof=1)) < 1e-5
    win = _rolling_stat_past_only(t, x, np.zeros(x.size, dtype=np.int64), "50", "std")
    assert abs(win[700] - x[650:700].std(ddof=1)) < 1e-5


def test_inter_event_expanding_std_large_offset_matches_reference():
    """Entity inter-event expanding population std must stay finite and accurate on epoch-scale values (old form returned a clipped garbage value)."""
    x = _epoch_like()
    _means, stds, _med = _expanding_stats_njit(x, np.array([0], dtype=np.int64), np.array([x.size], dtype=np.int64))
    assert abs(stds[-1] - x.std()) < 1e-5
    assert abs(stds[300] - x[:301].std()) < 1e-5


def test_anchor_linear_extrapolation_is_exact_without_additive_pad():
    """On an exactly linear anchor series the leave-one-out residual is exactly zero; an additive 1e-12 pad on the slope denominator left ~6e-12."""
    n = 30
    label = 3.0 * np.arange(n, dtype=np.float64)
    out = anchor_residual_rmse_features(label, np.ones(n, dtype=bool), K_slope=2, K_rmse=3)
    resid = np.asarray(out["anchor_loo_residual"])
    assert np.nanmax(np.abs(resid)) < 1e-13


def test_hurst_family_is_scale_invariant_down_to_tiny_amplitudes():
    """DFA and Higuchi exponents of a series must not depend on its overall scale; an additive 1e-12 pad on log-fluctuations broke this below ~1e-9."""
    x = np.random.default_rng(1).normal(size=400)
    for fn in (dfa_alpha, higuchi_fd, dfa_alpha2_quadratic):
        ref = fn(x)
        assert abs(fn(x * 1e-11) - ref) < 1e-9, fn.__name__
        assert abs(fn(x * 1e-100) - ref) < 1e-9, fn.__name__
    assert dfa_alpha(np.ones(400)) == 0.0


def test_consensus_entropy_is_scale_invariant_for_tiny_predictions():
    """Histogram binning of per-row predictions uses the row range itself, so 1e-10-scale predictions bin exactly like unit-scale ones."""
    base = np.random.default_rng(0).uniform(0.0, 1.0, (200, 6))
    unit = predictor_consensus_entropy(base)
    tiny = predictor_consensus_entropy(base * 1e-10)
    assert np.array_equal(unit, tiny)
    assert float(unit.mean()) > 1.0


def test_kalman_filter_is_scale_equivariant_for_tiny_variances():
    """Filtering data and sigmas scaled by 1e-8 gives the same relative tracking error; the pad on the innovation variance used to kill the Kalman gain."""
    rng = np.random.default_rng(0)
    obs = np.cumsum(rng.normal(0, 1, 200)) + rng.normal(0, 1, 200)
    errs = []
    for sc in (1.0, 1e-8):
        res = kalman_filter_posterior_1d(obs * sc, transition_sigma=sc, observation_sigma=sc, initial_variance=sc * sc)
        errs.append(float(np.mean(np.abs(res["mean"] - obs * sc)) / sc))
    assert abs(errs[0] - errs[1]) < 1e-6


def test_spectral_centroid_is_scale_invariant_for_tiny_amplitudes():
    """The spectral centroid is a ratio of power sums, so a 1e-8 amplitude must give the same centroid; the 1e-12 pad on total power crushed it."""
    rng = np.random.default_rng(0)
    x = np.sin(np.arange(300) * 0.7) + rng.normal(0, 0.1, 300)
    g = np.zeros(300, dtype=np.int64)
    ref = rolling_spectral_centroid(x, g, window_K=32)
    tiny = rolling_spectral_centroid(x * 1e-8, g, window_K=32)
    assert np.allclose(ref, tiny, rtol=1e-9, equal_nan=True)


def test_safe_div_zero_denominator_and_nan_propagation():
    """safe_div returns 0 only for an exactly zero denominator, divides tiny denominators exactly, and propagates NaN."""
    out = safe_div(np.array([1.0, 2e-20, np.nan, 3.0]), np.array([0.0, 4e-20, 1.0, np.nan]))
    assert out[0] == 0.0
    assert out[1] == 0.5
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_numpy_quantile_binner_matches_njit_binner_on_ties():
    """The numpy binner must break ties like the stable njit binner; the unstable default sort moved ~75% of rows on a 5-value column."""
    rng = np.random.default_rng(0)
    for card in (5, 2000):
        x = rng.integers(0, card, 20000).astype(np.float64)
        assert np.array_equal(_quantile_bin_njit(x, 20), _quantile_bin_numpy(x, 20))


def test_class_balanced_mi_with_unit_weights_equals_plain_mi_on_ties():
    """With all class weights 1 the balanced MI kernel must equal the plain plug-in MI even on tied columns (needs the same stable tie order)."""
    rng = np.random.default_rng(0)
    n = 20000
    X = rng.integers(0, 5, size=(n, 3)).astype(np.float64)
    y = (rng.random(n) < 0.3).astype(np.int64)
    plain = _plugin_mi_classif_batch_njit(X, y, 20)
    balanced = _class_balanced_mi_batch_njit(X, y, np.ones(2), 20)
    assert np.allclose(plain, balanced, rtol=0.0, atol=1e-12)


def test_composite_ratio_feature_survives_tiny_scale_column():
    """The ratio-to-group-mean feature of a 1e-14-scale column must vary; an absolute 1e-12 mean guard turned it into a constant 1.0."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"g": rng.integers(0, 4, n).astype(str), "h": rng.integers(0, 3, n).astype(str), "v": (5.0 + rng.normal(0, 1, n)) * 1e-14})
    enc, _ = generate_composite_group_agg_features(X, [["g", "h"]], ["v"])
    ratio = enc[next(c for c in enc.columns if "ratio" in c)]
    assert float(ratio.std()) > 0.1
    X_unit = X.assign(v=X["v"] * 1e14)
    enc_unit, _ = generate_composite_group_agg_features(X_unit, [["g", "h"]], ["v"])
    ratio_unit = enc_unit[next(c for c in enc_unit.columns if "ratio" in c)]
    assert np.allclose(ratio.to_numpy(), ratio_unit.to_numpy(), rtol=1e-9)


def test_shrunk_covariance_mean_accumulates_in_float64():
    """The class mean of a float32 block with a large offset must be accurate; float32 axis-0 accumulation lost ~1.3% over 2e6 rows."""
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(2_000_000, 3)) + 100.0).astype(np.float32)
    mean, _inv = _shrunk_covariance(X)
    assert np.allclose(mean, X.astype(np.float64).mean(axis=0), rtol=1e-6)
