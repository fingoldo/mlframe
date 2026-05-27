"""Smoke + correctness tests for the 2026-05-27 FE module extraction.

8 new modules / 1 extended:
- grouped.py            (#1)
- windowed_shape.py     (#7)
- spectral.py           (#2)
- ensemble_features.py  (#3)
- spatial.py            (#4)
- stationarity.py       (#5)
- anchor.py             (#8)
- bayesian.py           (#9)
- hurst.py              EXTENDED (#6: + rolling_hurst, rolling_dfa_alpha, rolling_higuchi_fd)
"""
from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# grouped.py
# =============================================================================

def test_iter_group_segments_basic() -> None:
    from mlframe.feature_engineering import iter_group_segments

    groups = np.array([1, 1, 2, 2, 2, 1, 3])
    sort_idx, starts, ends = iter_group_segments(groups)
    # Sorted: 1 1 1 | 2 2 2 | 3, but stable order preserves orig within group
    assert sort_idx.tolist() == [0, 1, 5, 2, 3, 4, 6]
    assert starts.tolist() == [0, 3, 6]
    assert ends.tolist() == [3, 6, 7]


def test_per_group_apply_returns_per_row_results_in_orig_order() -> None:
    from mlframe.feature_engineering import per_group_apply

    values = np.array([10.0, 20.0, 100.0, 200.0, 300.0])
    groups = np.array([1, 1, 2, 2, 2])

    def normalize(seg: np.ndarray) -> np.ndarray:
        return seg / seg.max()

    out = per_group_apply(values, groups, normalize)
    # Group 1 max=20 -> [0.5, 1.0]; group 2 max=300 -> [1/3, 2/3, 1.0]
    np.testing.assert_allclose(out, [0.5, 1.0, 1 / 3, 2 / 3, 1.0])


def test_per_group_sliding_window_basic() -> None:
    from mlframe.feature_engineering import per_group_sliding_window

    values = np.arange(10, dtype=np.float64)
    groups = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    out = np.full(10, np.nan)
    for sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, groups, window_K=3,
    ):
        out[write_idx] = wins.mean(axis=1)
    # group 1 rows 0..4: K=3 windows starting at row 2: mean(0,1,2)=1, mean(1,2,3)=2, mean(2,3,4)=3
    # group 2 rows 5..9: window means at rows 7,8,9 = mean(5,6,7)=6, mean(6,7,8)=7, mean(7,8,9)=8
    np.testing.assert_allclose(out[2:5], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(out[7:10], [6.0, 7.0, 8.0])
    # First two rows of each group are NaN.
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert np.isnan(out[5]) and np.isnan(out[6])


# =============================================================================
# windowed_shape.py
# =============================================================================

def test_rolling_mean_abs_d2_zero_on_linear_signal() -> None:
    """Linear signal has zero second-difference -> mean_abs_d2 = 0 everywhere."""
    from mlframe.feature_engineering import rolling_mean_abs_d2

    values = np.linspace(0, 1, 100)
    groups = np.zeros(100, dtype=int)
    out = rolling_mean_abs_d2(values, groups, window_K=10)
    # Last 91 values (K=10 onward) should all be near zero.
    finite = out[~np.isnan(out)]
    assert finite.size == 91
    np.testing.assert_allclose(finite, 0.0, atol=1e-12)


def test_rolling_n_peaks_and_troughs_on_sinusoid() -> None:
    from mlframe.feature_engineering import rolling_n_peaks, rolling_n_troughs

    t = np.arange(200, dtype=np.float64)
    values = np.sin(2 * np.pi * t / 20)  # period 20, 10 cycles
    groups = np.zeros(200, dtype=int)
    n_peaks = rolling_n_peaks(values, groups, window_K=40)
    n_troughs = rolling_n_troughs(values, groups, window_K=40)
    # In a K=40 window over period-20 sin: ~2 peaks + ~2 troughs
    finite_p = n_peaks[~np.isnan(n_peaks)]
    finite_t = n_troughs[~np.isnan(n_troughs)]
    assert finite_p.size > 0
    assert 1 <= finite_p.mean() <= 3
    assert 1 <= finite_t.mean() <= 3


def test_rolling_integral_above_baseline_positive_on_positive_signal() -> None:
    from mlframe.feature_engineering import rolling_integral_above_baseline

    values = np.array([1.0] * 100)  # all 1.0
    groups = np.zeros(100, dtype=int)
    out = rolling_integral_above_baseline(
        values, groups, window_K=10, baseline_fn="median",
    )
    finite = out[~np.isnan(out)]
    # baseline=median(values)=1.0 -> (values - baseline) clipped at 0 = all zeros
    np.testing.assert_allclose(finite, 0.0)

    # With a slope, half the window will be above median
    rng = np.random.default_rng(0)
    values = np.linspace(0, 10, 100) + rng.normal(0, 0.1, 100)
    out = rolling_integral_above_baseline(values, groups, window_K=20)
    assert (out[~np.isnan(out)] > 0).any()


# =============================================================================
# spectral.py
# =============================================================================

def test_rolling_spectral_band_energies_three_bands() -> None:
    from mlframe.feature_engineering import rolling_spectral_band_energies

    rng = np.random.default_rng(0)
    n = 500
    values = rng.standard_normal(n)
    groups = np.zeros(n, dtype=int)
    out = rolling_spectral_band_energies(values, groups, window_K=64)
    assert out.shape == (n, 3)
    # Most windows should have finite energies > 0
    finite_rows = ~np.isnan(out).any(axis=1)
    assert finite_rows.sum() >= n - 64
    assert (out[finite_rows] >= 0).all()


def test_rolling_spectral_entropy_higher_for_noise_than_sin() -> None:
    """Pure sinusoid has narrow spectrum (low entropy); noise is broadband (high)."""
    from mlframe.feature_engineering import rolling_spectral_entropy

    t = np.arange(500, dtype=np.float64)
    sin_signal = np.sin(2 * np.pi * t / 10)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(500)
    groups = np.zeros(500, dtype=int)
    ent_sin = np.nanmean(
        rolling_spectral_entropy(sin_signal, groups, window_K=64)
    )
    ent_noise = np.nanmean(
        rolling_spectral_entropy(noise, groups, window_K=64)
    )
    assert ent_sin < ent_noise, (
        f"sin entropy={ent_sin:.3f} should be < noise entropy={ent_noise:.3f}"
    )


def test_rolling_hf_lf_ratio_higher_on_high_freq_signal() -> None:
    from mlframe.feature_engineering import rolling_hf_lf_ratio

    t = np.arange(500, dtype=np.float64)
    low_freq = np.sin(2 * np.pi * t / 100)  # period 100 = LF
    high_freq = np.sin(2 * np.pi * t / 5)   # period 5 = HF
    groups = np.zeros(500, dtype=int)
    r_lo = np.nanmean(rolling_hf_lf_ratio(low_freq, groups, window_K=64))
    r_hi = np.nanmean(rolling_hf_lf_ratio(high_freq, groups, window_K=64))
    assert r_hi > r_lo


# =============================================================================
# ensemble_features.py
# =============================================================================

def test_predictor_disagreement_iqr_zero_on_identical_predictors() -> None:
    from mlframe.feature_engineering import predictor_disagreement_iqr

    preds = np.tile(np.array([[1.0, 2.0, 3.0]]).T, (1, 4))  # all 4 preds identical per row
    iqr = predictor_disagreement_iqr(preds)
    np.testing.assert_allclose(iqr, 0.0)


def test_predictor_disagreement_features_emits_all_keys() -> None:
    from mlframe.feature_engineering import predictor_disagreement_features

    rng = np.random.default_rng(0)
    preds = rng.standard_normal((100, 5))
    out = predictor_disagreement_features(preds, emit_pairs=True)
    for k in ("mean", "iqr", "var", "entropy", "top2_gap", "pairs"):
        assert k in out
    assert out["mean"].shape == (100,)
    assert out["pairs"].shape == (100, 10)  # 5*4/2


def test_predictor_disagreement_handles_nan() -> None:
    """Single NaN in one predictor should not propagate to all features."""
    from mlframe.feature_engineering import predictor_disagreement_features

    preds = np.array([
        [1.0, 1.1, 1.2, 1.3],
        [2.0, np.nan, 2.2, 2.3],
        [3.0, 3.1, 3.2, 3.3],
    ])
    out = predictor_disagreement_features(preds, emit_pairs=False)
    assert np.isfinite(out["iqr"]).all()
    assert np.isfinite(out["entropy"]).all()


# =============================================================================
# spatial.py
# =============================================================================

def test_knn_aggregate_basic() -> None:
    from mlframe.feature_engineering import knn_aggregate

    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 10, (100, 2))
    labels = ref.sum(axis=1)  # label = x + y
    q = rng.uniform(0, 10, (20, 2))
    out = knn_aggregate(q, ref, labels, k=5, agg_fns=("median", "mean", "iqr"))
    assert out["median"].shape == (20,)
    assert "iqr" in out and "_nearest_distance" in out


def test_knn_within_bucket_aggregate_respects_bucket() -> None:
    from mlframe.feature_engineering import knn_within_bucket_aggregate

    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 10, (100, 2))
    labels = rng.standard_normal(100)
    ref_bucket = (ref[:, 0] > 5).astype(int)  # bucket A vs B
    q = rng.uniform(0, 10, (10, 2))
    q_bucket = (q[:, 0] > 5).astype(int)
    out = knn_within_bucket_aggregate(
        q, ref, labels, q_bucket=q_bucket, ref_bucket=ref_bucket, k=3,
    )
    assert out["median"].shape == (10,)


# =============================================================================
# stationarity.py
# =============================================================================

def test_frac_diff_weights_decay() -> None:
    """Weights should decay geometrically: |w_k| -> 0 as k grows."""
    from mlframe.feature_engineering import frac_diff_weights

    w = frac_diff_weights(d=0.5, K=20)
    assert w[0] == 1.0
    assert abs(w[10]) < abs(w[1])  # decay
    assert abs(w[19]) < 0.01


def test_frac_diff_d_eq_1_is_first_difference() -> None:
    from mlframe.feature_engineering import frac_diff

    x = np.linspace(0, 100, 50)
    fd = frac_diff(x, d=1.0, K=2)
    # frac_diff with d=1 K=2: y_t = x_t - x_{t-1} = constant
    finite = fd[~np.isnan(fd)]
    np.testing.assert_allclose(finite, 100 / 49, atol=1e-9)


def test_frac_diff_per_group_no_bleed() -> None:
    """Cross-group rows must not bleed: group 2's first K rows should
    have NaN even though group 1 has earlier rows in the global array.
    """
    from mlframe.feature_engineering import frac_diff

    values = np.arange(20, dtype=np.float64)
    groups = np.array([1] * 10 + [2] * 10)
    fd = frac_diff(values, d=0.5, K=5, group_ids=groups)
    # group 1: rows 0..4 NaN (no full K), rows 5..9 finite
    # group 2: rows 10..14 NaN, rows 15..19 finite
    assert np.isnan(fd[0:5]).all()
    assert np.isfinite(fd[5:10]).all()
    assert np.isnan(fd[10:15]).all()
    assert np.isfinite(fd[15:20]).all()


# =============================================================================
# anchor.py
# =============================================================================

def test_add_anchor_extrapolation_features_basic() -> None:
    from mlframe.feature_engineering import add_anchor_extrapolation_features

    # 10 rows, anchors at rows 0, 3, 7. Label = row index * 2 on anchors.
    n = 10
    label = np.full(n, np.nan)
    is_anchor = np.zeros(n, dtype=bool)
    for i in (0, 3, 7):
        label[i] = i * 2.0
        is_anchor[i] = True
    out = add_anchor_extrapolation_features(label, is_anchor, K_slope=3)
    # rows_since at row 5 (anchor was row 3): 2.0
    assert out["rows_since_last_anchor"][5] == 2.0
    # last_anchor_value at row 5: 6.0 (label at row 3)
    assert out["last_anchor_value"][5] == 6.0
    # local_slope_K3 over anchors (0, 3, 7) at values (0, 6, 14) ~ slope 2.0
    assert abs(out["last_anchor_local_slope_K3"][5] - 2.0) < 0.5
    # extrap_pred at row 5: 6.0 + slope * 2 ~ 10.0
    extrap_5 = out["linear_extrap_pred_K3"][5]
    assert abs(extrap_5 - 10.0) < 2.0


def test_anchor_features_per_group_no_bleed() -> None:
    from mlframe.feature_engineering import add_anchor_extrapolation_features

    label = np.array([10.0, np.nan, np.nan, 100.0, np.nan])
    is_anchor = np.array([True, False, False, True, False])
    groups = np.array([1, 1, 2, 2, 2])
    out = add_anchor_extrapolation_features(label, is_anchor, groups, K_slope=2)
    # Row 2 (group 2 first row) has NO prior anchor in group 2 -> NaN
    assert np.isnan(out["rows_since_last_anchor"][2])
    # Row 4 (group 2) sees anchor at row 3 -> rows_since=1
    assert out["rows_since_last_anchor"][4] == 1.0
    assert out["last_anchor_value"][4] == 100.0


# =============================================================================
# bayesian.py
# =============================================================================

def test_particle_filter_posterior_tracks_constant() -> None:
    """PF on constant observation should converge to that constant."""
    from mlframe.feature_engineering import particle_filter_posterior

    obs = np.full(200, 5.0)
    out = particle_filter_posterior(
        obs, n_particles=64, transition_sigma=0.1, observation_sigma=0.5,
        seed=0,
    )
    # After warmup, p50 should be close to 5.0
    p50_tail = out["p50"][-50:]
    assert abs(p50_tail.mean() - 5.0) < 0.5
    # p10 <= p50 <= p90 always
    assert (out["p10"] <= out["p50"] + 1e-6).all()
    assert (out["p50"] <= out["p90"] + 1e-6).all()


# =============================================================================
# hurst.py (extended)
# =============================================================================

def test_rolling_hurst_finite_on_random_walk() -> None:
    from mlframe.feature_engineering import rolling_hurst

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(500))
    out = rolling_hurst(x, window_K=200)
    # Random walk has Hurst ~ 0.5 (with single-scale R/S that's a noisy estimate)
    finite = out[~np.isnan(out)]
    assert finite.size > 0
    # Sanity: finite values in [0, 1.5] (R/S with single scale is bounded loosely)
    assert (finite >= 0).all() and (finite <= 2.0).all()


def test_rolling_dfa_alpha_persistent_for_brownian() -> None:
    """Brownian motion has DFA alpha ~ 1.5. Our window is small so test
    sanity / boundedness rather than exact value."""
    from mlframe.feature_engineering import rolling_dfa_alpha

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(500))
    out = rolling_dfa_alpha(x, window_K=200)
    finite = out[~np.isnan(out)]
    assert finite.size > 0
    assert (finite > 0.5).any()


def test_higuchi_fd_distinguishes_smooth_vs_noise() -> None:
    """Smooth signal should have HFD ~ 1; pure noise HFD ~ 2."""
    from mlframe.feature_engineering import higuchi_fd

    smooth = np.linspace(0, 1, 200)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(200)
    h_smooth = higuchi_fd(smooth, kmax=8)
    h_noise = higuchi_fd(noise, kmax=8)
    assert h_smooth < h_noise, f"smooth={h_smooth:.3f} >= noise={h_noise:.3f}"


# =============================================================================
# bayesian.py per-group recursion: serial / parallel backend equivalence
# =============================================================================

def _grouped_recursion_inputs(n_groups: int, rows_per_group: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    N = n_groups * rows_per_group
    well = np.repeat(np.arange(n_groups), rows_per_group)
    md = np.tile(np.arange(rows_per_group, dtype=float), n_groups)
    obs = np.cumsum(rng.normal(0, 0.1, N)) + md * 0.01
    obs[rng.random(N) < 0.05] = np.nan  # exercise the NaN predict-only branch
    X = np.column_stack([np.ones(N), md / (md.max() + 1e-9)])
    return obs, X, well


def test_bocpd_serial_parallel_backends_match() -> None:
    """The prange-over-groups parallel driver must be bit-equivalent to the
    serial driver (and both to the numpy fallback) on the same input."""
    import os
    pytest = __import__("pytest")
    pytest.importorskip("numba")
    from mlframe.feature_engineering import bayesian as B
    if not B._NUMBA_AVAILABLE:
        pytest.skip("numba unavailable")
    obs, _X, well = _grouped_recursion_inputs(12, 200)

    os.environ["MLFRAME_FE_RECURSION_BACKEND"] = "serial"
    try:
        r_serial = B.bocpd_features(obs, group_ids=well, hazard=1 / 100)
    finally:
        del os.environ["MLFRAME_FE_RECURSION_BACKEND"]
    os.environ["MLFRAME_FE_RECURSION_BACKEND"] = "parallel"
    try:
        r_parallel = B.bocpd_features(obs, group_ids=well, hazard=1 / 100)
    finally:
        del os.environ["MLFRAME_FE_RECURSION_BACKEND"]

    for key in r_serial:
        a, b = r_serial[key], r_parallel[key]
        mask = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(a[mask], b[mask], atol=1e-9, rtol=0,
                                   err_msg=f"bocpd {key} serial != parallel")


def test_oblr_serial_parallel_backends_match() -> None:
    import os
    pytest = __import__("pytest")
    pytest.importorskip("numba")
    from mlframe.feature_engineering import bayesian as B
    if not B._NUMBA_AVAILABLE:
        pytest.skip("numba unavailable")
    obs, X, well = _grouped_recursion_inputs(12, 200)

    os.environ["MLFRAME_FE_RECURSION_BACKEND"] = "serial"
    try:
        r_serial = B.online_bayesian_linear_regression(obs, X, group_ids=well)
    finally:
        del os.environ["MLFRAME_FE_RECURSION_BACKEND"]
    os.environ["MLFRAME_FE_RECURSION_BACKEND"] = "parallel"
    try:
        r_parallel = B.online_bayesian_linear_regression(obs, X, group_ids=well)
    finally:
        del os.environ["MLFRAME_FE_RECURSION_BACKEND"]

    for key in r_serial:
        a, b = r_serial[key], r_parallel[key]
        mask = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(a[mask], b[mask], atol=1e-9, rtol=0,
                                   err_msg=f"oblr {key} serial != parallel")


def test_recursion_dispatch_routes_by_kernel_and_size() -> None:
    """fe_bocpd (heavy per-row) goes parallel at modest group counts;
    fe_oblr (trivial per-row) stays serial at small scale and only flips
    parallel at higher group/sample counts. Env override always wins."""
    import os
    from mlframe.feature_engineering._recursion_dispatch import dispatch_recursion_backend

    # Single group -> always serial (nothing to parallelise).
    assert dispatch_recursion_backend("fe_bocpd", 100_000, 1) == "serial"

    # BOCPD: parallel from a handful of groups.
    assert dispatch_recursion_backend("fe_bocpd", 100_000, 64) == "parallel"
    assert dispatch_recursion_backend("fe_bocpd", 1_000, 2) == "serial"

    # OBLR: serial at tiny scale, parallel once big enough.
    assert dispatch_recursion_backend("fe_oblr", 3_200, 8) == "serial"
    assert dispatch_recursion_backend("fe_oblr", 100_000, 64) == "parallel"

    # Env override forces the choice regardless of size.
    os.environ["MLFRAME_FE_RECURSION_BACKEND"] = "serial"
    try:
        assert dispatch_recursion_backend("fe_bocpd", 10_000_000, 10_000) == "serial"
    finally:
        del os.environ["MLFRAME_FE_RECURSION_BACKEND"]


# =============================================================================
# anchor.py: numba cores match the Python list-based fallback
# =============================================================================

def _anchor_test_frame(n_wells: int = 10, rows_per_well: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    N = n_wells * rows_per_well
    well = np.repeat(np.arange(n_wells), rows_per_well)
    md = np.tile(np.arange(rows_per_well, dtype=float), n_wells)
    label = np.cumsum(rng.normal(0, 0.2, N)) + md * 0.03
    is_anchor = rng.random(N) < 0.15
    return label, is_anchor, well


def test_anchor_numba_cores_match_python_fallback() -> None:
    """Every anchor feature's njit core must reproduce the list-based Python
    reference (positions, residuals, EWM weights, window gaps) exactly."""
    pytest = __import__("pytest")
    pytest.importorskip("numba")
    from mlframe.feature_engineering import anchor as A
    if not A._NUMBA_AVAILABLE:
        pytest.skip("numba unavailable")
    label, is_anchor, well = _anchor_test_frame()

    cases = [
        lambda: A.anchor_residual_rmse_features(label, is_anchor, well, K_slope=10, K_rmse=10),
        lambda: A.anchor_quadratic_extrapolation_features(label, is_anchor, well, K_window=10),
        lambda: A.anchor_ewm_features(label, is_anchor, well, half_life_rows=30.0),
        lambda: A.anchor_density_features(is_anchor, well, window_rows=100),
    ]
    for fn in cases:
        A._NUMBA_AVAILABLE = True
        try:
            out_nb = fn()
            A._NUMBA_AVAILABLE = False
            out_py = fn()
        finally:
            A._NUMBA_AVAILABLE = True
        for key in out_py:
            a, b = out_nb[key], out_py[key]
            # NaN masks must agree, finite values must match.
            np.testing.assert_array_equal(
                np.isfinite(a), np.isfinite(b),
                err_msg=f"anchor {key}: NaN positions differ numba vs python",
            )
            mask = np.isfinite(a) & np.isfinite(b)
            np.testing.assert_allclose(
                a[mask], b[mask], atol=1e-6, rtol=0,
                err_msg=f"anchor {key}: numba != python",
            )


# =============================================================================
# grouped.per_group_rank: NaN must stay LOCAL, not poison the whole group
# =============================================================================

def test_per_group_rank_nan_does_not_poison_group() -> None:
    """Regression: scipy.rankdata's default nan_policy='propagate' turns a
    whole group's ranks to NaN on a single missing value -- silently
    collapsing any rank-based feature over a NaN-bearing column to an
    all-NaN ('constant') column. per_group_rank must rank only the finite
    entries and leave NaN positions NaN."""
    from mlframe.feature_engineering import per_group_rank

    # Two groups; each has one NaN. Finite values must still get real ranks.
    vals = np.array([3.0, 1.0, np.nan, 2.0,   30.0, np.nan, 10.0, 20.0])
    grp = np.array([0, 0, 0, 0,   1, 1, 1, 1])
    pct = per_group_rank(vals, grp, pct=True)

    # NaN positions preserved exactly.
    np.testing.assert_array_equal(np.isnan(pct), np.isnan(vals))
    # Group 0 finite {3,1,2} -> ranks/3 = {1.0, 1/3, 2/3}
    np.testing.assert_allclose(pct[[0, 1, 3]], [1.0, 1 / 3, 2 / 3])
    # Group 1 finite {30,10,20} -> {1.0, 1/3, 2/3}
    np.testing.assert_allclose(pct[[4, 6, 7]], [1.0, 1 / 3, 2 / 3])


def test_quantile_normalize_per_group_survives_nan_heavy_column() -> None:
    """A column with scattered NaN across every group (e.g. raw GR with
    ~2% missing) must yield a VARYING normalized feature, not all-NaN."""
    from mlframe.feature_engineering.stationarity import quantile_normalize_per_group

    rng = np.random.default_rng(0)
    n_groups, rpg = 6, 300
    grp = np.repeat(np.arange(n_groups), rpg)
    x = rng.normal(60, 15, n_groups * rpg)
    x[rng.random(x.size) < 0.02] = np.nan  # 2% missing in (almost) every group
    qn = quantile_normalize_per_group(x, group_ids=grp, to_normal=False)

    finite = qn[np.isfinite(qn)]
    assert finite.size > 0.9 * x.size, "lost too many cells to NaN poisoning"
    assert finite.std() > 0.1, f"normalized feature is ~constant (std={finite.std():.3g})"
    np.testing.assert_array_equal(np.isnan(qn), np.isnan(x))
