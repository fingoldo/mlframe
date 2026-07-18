"""Unit tests for ``mlframe.training._cv_aggregation``.

Covers:
  - all 5 aggregator modes
  - direction sign inversion (min vs max)
  - auto-flip of quantile by direction
  - t-LCB reference against scipy.stats.t
  - correlation_inflation applied
  - Pareto frontier construction (min and max direction)
  - select_from_pareto with risk_quantile knob
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.training._cv_aggregation import (
    AUTO_INFLATION,
    aggregate_fold_scores,
    compute_pareto_frontier,
    nadeau_bengio_inflation,
    select_from_pareto,
)


def test_aggregate_mean_baseline() -> None:
    """Aggregate mean baseline."""
    assert aggregate_fold_scores([1.0, 2.0, 3.0], mode="mean") == pytest.approx(2.0)


def test_aggregate_mean_minus_std_picks_stable() -> None:
    """Aggregate mean minus std picks stable."""
    stable = aggregate_fold_scores([1.0, 1.0, 1.0, 1.0], mode="mean_minus_std", alpha=1.0, direction="min")
    unstable = aggregate_fold_scores([0.8, 0.8, 0.8, 1.8], mode="mean_minus_std", alpha=1.0, direction="min")
    assert stable == pytest.approx(1.0)
    # unstable mean = 1.05, std(ddof=1) = 0.5; penalty pushes UP for direction='min'
    assert unstable > stable
    # Same scores under direction='max' should give the OPPOSITE ordering:
    stable_max = aggregate_fold_scores([1.0, 1.0, 1.0, 1.0], mode="mean_minus_std", alpha=1.0, direction="max")
    unstable_max = aggregate_fold_scores([0.8, 0.8, 0.8, 1.8], mode="mean_minus_std", alpha=1.0, direction="max")
    assert unstable_max < stable_max, "for max-metrics, penalty pushes DOWN; unstable should look worse (smaller)"


def test_aggregate_t_lcb_matches_scipy() -> None:
    """Aggregate t lcb matches scipy."""
    from scipy.stats import t as _t

    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    arr = np.asarray(scores)
    k = len(scores)
    se = float(np.std(arr, ddof=1) / math.sqrt(k))
    tq = float(_t.ppf(0.9, df=k - 1))

    got = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9)
    expected = float(np.mean(arr)) + tq * se
    assert got == pytest.approx(expected, rel=1e-12)


def test_aggregate_correlation_inflation_applied() -> None:
    """Aggregate correlation inflation applied."""
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    naive = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, correlation_inflation=1.0)
    inflated = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, correlation_inflation=1.5)
    # Larger penalty (inflated std) -> for direction='min', the augmented score is LARGER
    assert inflated > naive
    # Same effect via mean_minus_std:
    naive2 = aggregate_fold_scores(scores, mode="mean_minus_std", direction="min", alpha=1.0, correlation_inflation=1.0)
    inflated2 = aggregate_fold_scores(scores, mode="mean_minus_std", direction="min", alpha=1.0, correlation_inflation=1.5)
    assert inflated2 > naive2


def test_aggregate_quantile_auto_flip() -> None:
    # 10 scores 0.1, 0.2, ..., 1.0. quantile_level=0.9 -> upper for min (around 0.91), lower for max (around 0.19)
    """Aggregate quantile auto flip."""
    scores = [round(0.1 * i, 1) for i in range(1, 11)]
    upper = aggregate_fold_scores(scores, mode="quantile", direction="min", quantile_level=0.9)
    lower = aggregate_fold_scores(scores, mode="quantile", direction="max", quantile_level=0.9)
    assert upper > 0.85, f"min-direction must read upper tail; got {upper}"
    assert lower < 0.25, f"max-direction must read lower tail; got {lower}"


def test_aggregate_quantile_robust_to_one_outlier() -> None:
    # 9 stable + 1 outlier. quantile@0.8 stable; mean shifted.
    """Aggregate quantile robust to one outlier."""
    scores = [1.0] * 9 + [10.0]
    q = aggregate_fold_scores(scores, mode="quantile", direction="min", quantile_level=0.8)
    mean_penalty = aggregate_fold_scores(scores, mode="mean_minus_std", direction="min", alpha=1.0)
    assert q <= 2.0, f"quantile@0.8 should ignore the single outlier; got {q}"
    assert mean_penalty > q, f"mean+std is dominated by outlier; got mean_penalty={mean_penalty}"


def test_aggregate_median_mad_outlier_resilient() -> None:
    # one fold outlier doesn't shift median; mean does.
    """Aggregate median mad outlier resilient."""
    scores = [1.0, 1.0, 1.0, 1.0, 10.0]
    med = aggregate_fold_scores(scores, mode="median_minus_mad", direction="min", alpha=1.0)
    mean_pen = aggregate_fold_scores(scores, mode="mean_minus_std", direction="min", alpha=1.0)
    assert med < mean_pen, f"median+MAD must be lower (more robust); got med={med} mean={mean_pen}"


def test_aggregate_direction_sign_inversion() -> None:
    """Aggregate direction sign inversion."""
    scores = [0.5, 0.6, 0.7]
    s_min = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9)
    s_max = aggregate_fold_scores(scores, mode="t_lcb", direction="max", confidence=0.9)
    mean = float(np.mean(scores))
    assert s_min > mean, "min-direction LCB adds penalty"
    assert s_max < mean, "max-direction LCB subtracts penalty"
    assert (s_min - mean) == pytest.approx(mean - s_max, rel=1e-9), "sign-symmetric around mean"


def test_aggregate_unknown_mode_raises() -> None:
    """Aggregate unknown mode raises."""
    with pytest.raises(ValueError, match="unknown mode"):
        aggregate_fold_scores([1.0, 2.0], mode="bogus")  # type: ignore[arg-type]


def test_aggregate_single_fold_falls_back_to_mean() -> None:
    # K=1 can't produce std/quantile meaningfully; just return the value.
    """Aggregate single fold falls back to mean."""
    assert aggregate_fold_scores([0.42], mode="t_lcb") == pytest.approx(0.42)
    assert aggregate_fold_scores([0.42], mode="quantile") == pytest.approx(0.42)
    assert aggregate_fold_scores([0.42], mode="mean_minus_std") == pytest.approx(0.42)


def test_pareto_frontier_min_direction() -> None:
    # (mean, std) points: best mean = 0.8, best std = 0.3
    """Pareto frontier min direction."""
    pts = [(1.0, 0.5), (0.9, 0.6), (0.8, 0.7), (0.85, 0.4), (1.1, 0.3)]
    front = compute_pareto_frontier(pts, mean_direction="min")
    # Expected: 2 (best mean), 3 (mid mean + lower std), 4 (worst mean but lowest std)
    assert front == [2, 3, 4]


def test_pareto_frontier_max_direction() -> None:
    # max-direction: higher mean is better; still want lower std
    """Pareto frontier max direction."""
    pts = [(1.0, 0.5), (0.9, 0.6), (1.2, 0.7), (1.15, 0.4), (0.7, 0.3)]
    front = compute_pareto_frontier(pts, mean_direction="max")
    # Sort by -mean: 2(1.2, 0.7), 3(1.15, 0.4), 0(1.0, 0.5), 1(0.9, 0.6), 4(0.7, 0.3)
    # keep when std improves: 2 (best_std=0.7), 3 (0.4 < 0.7), 4 (0.3 < 0.4)
    assert front == [2, 3, 4]


def test_pareto_frontier_equal_mean_excludes_dominated_higher_std() -> None:
    # Two points share the mean; the higher-std one is strictly dominated and must be dropped.
    # Pre-fix argsort kept both (it only rejected domination by a strictly-better-mean predecessor).
    """Pareto frontier equal mean excludes dominated higher std."""
    assert compute_pareto_frontier([(0.1, 0.5), (0.1, 0.2)], mean_direction="min") == [1]
    assert compute_pareto_frontier([(0.1, 0.2), (0.1, 0.5)], mean_direction="min") == [0]
    assert compute_pareto_frontier([(0.1, 0.5), (0.1, 0.2)], mean_direction="max") == [1]


def test_pareto_frontier_empty() -> None:
    """Pareto frontier empty."""
    assert compute_pareto_frontier([], mean_direction="min") == []


def test_pareto_frontier_single() -> None:
    """Pareto frontier single."""
    assert compute_pareto_frontier([(0.5, 0.1)], mean_direction="min") == [0]


def test_pareto_frontier_invalid_shape() -> None:
    """Pareto frontier invalid shape."""
    with pytest.raises(ValueError, match="expected"):
        compute_pareto_frontier([(0.5,)], mean_direction="min")  # type: ignore[list-item]


def test_select_from_pareto_risk_quantile_monotone() -> None:
    # Three iterations on the frontier; per-shard scores have known dispersion.
    """Select from pareto risk quantile monotone."""
    frontier = [0, 1, 2]
    iter_means = [1.0, 0.95, 1.05]
    iter_stds = [0.05, 0.20, 0.02]
    iter_shard_scores = [
        [0.95, 1.0, 1.05],  # tight cluster, mean 1.0
        [0.75, 0.95, 1.15],  # wide cluster, mean 0.95
        [1.03, 1.05, 1.07],  # very tight, mean 1.05
    ]
    # direction='min': aggressive (q=0.5) prefers low mean -> pick iter 1
    pick_aggressive = select_from_pareto(frontier, iter_means, iter_stds, iter_shard_scores, risk_quantile=0.5, direction="min")
    # conservative (q=0.95) penalizes the wide cluster heavily -> pick iter 2 (tightest)
    pick_conservative = select_from_pareto(frontier, iter_means, iter_stds, iter_shard_scores, risk_quantile=0.95, direction="min")
    assert pick_aggressive == 1, f"aggressive should pick lowest-mean iter; got {pick_aggressive}"
    # conservative pick should NOT be the wide iter (1), and should differ from aggressive.
    assert pick_conservative != 1, f"conservative should avoid wide cluster; got {pick_conservative}"


def test_select_from_pareto_empty_raises() -> None:
    """Select from pareto empty raises."""
    with pytest.raises(ValueError, match="empty frontier"):
        select_from_pareto([], [], [], [], risk_quantile=0.9)


def test_select_from_pareto_falls_back_to_mean_std_when_shard_scores_missing() -> None:
    # Pre-fix: an iteration with no shard scores was skipped outright, ignoring iter_means/iter_stds
    # entirely. Post-fix: it competes via a normal-approximation risk quantile from mean/std, so a
    # frontier point that legitimately has no shard scores is not automatically excluded from selection.
    """Select from pareto falls back to mean std when shard scores missing."""
    frontier = [0, 1]
    iter_means = [0.5, 2.0]  # iter 0 is much better on the mean alone
    iter_stds = [0.01, 0.01]  # both tight, so mean should dominate the risk-quantile comparison
    iter_shard_scores = [[], []]  # neither iteration has shard scores -> both must use the fallback
    best = select_from_pareto(frontier, iter_means, iter_stds, iter_shard_scores, risk_quantile=0.9, direction="min")
    assert best == 0, f"fallback should still pick the iteration with the better mean/std; got {best}"


def test_nadeau_bengio_factor_matches_closed_form() -> None:
    # Standard K-fold: test_frac = 1/K, train_frac = (K-1)/K -> factor = sqrt(1 + K/(K-1)).
    """Nadeau bengio factor matches closed form."""
    for k in (2, 3, 5, 10, 20):
        got = nadeau_bengio_inflation(k, 1.0 / k)
        expected = math.sqrt(1.0 + k / (k - 1.0))
        assert got == pytest.approx(expected, rel=1e-12), f"K={k}"
    # K=5 standard-fold default is ~1.5 (the historical hardcoded SliceStableES value).
    assert nadeau_bengio_inflation(5, 0.2) == pytest.approx(1.5, abs=0.02)
    # Always >= 1.0 (it is an INFLATION; never deflates the interval).
    assert nadeau_bengio_inflation(10, 0.1) > 1.0


def test_nadeau_bengio_explicit_train_frac_overlap() -> None:
    # Overlapping resamples: train_frac can exceed 1 - test_frac (e.g. repeated 80/20 holdouts).
    # More train overlap relative to test -> SMALLER test_frac/train_frac -> SMALLER inflation.
    """Nadeau bengio explicit train frac overlap."""
    low_overlap = nadeau_bengio_inflation(5, test_frac=0.2, train_frac=0.8)
    high_overlap = nadeau_bengio_inflation(5, test_frac=0.2, train_frac=2.0)
    assert high_overlap < low_overlap
    assert high_overlap > 1.0


def test_nadeau_bengio_degenerate_returns_one() -> None:
    # K<2 has no dispersion; zero/negative fractions are degenerate. Never inflate.
    """Nadeau bengio degenerate returns one."""
    assert nadeau_bengio_inflation(1, 0.5) == 1.0
    assert nadeau_bengio_inflation(5, 0.0) == 1.0
    assert nadeau_bengio_inflation(5, 0.2, train_frac=0.0) == 1.0


def test_auto_inflation_without_geometry_is_naive() -> None:
    # The AUTO sentinel with NO split_geometry must be bit-identical to the explicit naive 1.0,
    # so the composite callers that pass neither stay unchanged.
    """Auto inflation without geometry is naive."""
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    auto = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9)
    naive = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, correlation_inflation=1.0)
    assert auto == pytest.approx(naive, rel=0, abs=0)
    # And the public default of correlation_inflation IS the sentinel.
    assert AUTO_INFLATION == "auto"


def test_auto_inflation_with_geometry_applies_nadeau_bengio() -> None:
    # With split_geometry supplied, AUTO resolves to the NB factor and widens the interval vs naive.
    """Auto inflation with geometry applies nadeau bengio."""
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    k = len(scores)
    naive = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, correlation_inflation=1.0)
    auto_geo = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, split_geometry=(k, 1.0 / k))
    explicit = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, correlation_inflation=nadeau_bengio_inflation(k, 1.0 / k))
    mean = float(np.mean(scores))
    # AUTO+geometry must equal the explicit NB-factor call (proves the resolution path).
    assert auto_geo == pytest.approx(explicit, rel=1e-12)
    # And it must be a strictly WIDER (less over-confident) one-sided interval than naive for direction='min'.
    assert (auto_geo - mean) > (naive - mean) > 0.0


def test_biz_val_cv_aggregation_nb_inflation_widens_interval_by_factor() -> None:
    """biz_value: the Nadeau-Bengio default must widen the t-LCB half-width by EXACTLY the NB factor.

    The whole point of M5 is that the CV-RMSE confidence intervals the cv_selector reads are over-confident when the
    folds share training rows. This pins the quantitative win: the half-width (penalty above the mean) of the
    NB-defaulted interval is the naive half-width times ``sqrt(1 + K * test_frac/train_frac)`` -- a ~1.5x widening at
    K=5. A regression that drops the factor back to 1.0 (the pre-M5 behaviour) fails this by ~33%."""
    scores = [1.0, 1.2, 0.9, 1.1, 1.05]  # K=5 RMSE-like fold scores
    k = len(scores)
    mean = float(np.mean(scores))
    naive = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, correlation_inflation=1.0)
    nb = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, split_geometry=(k, 1.0 / k))
    naive_halfwidth = naive - mean
    nb_halfwidth = nb - mean
    factor = nadeau_bengio_inflation(k, 1.0 / k)
    assert factor == pytest.approx(1.5, abs=0.02)
    # The widening is EXACTLY the NB factor (the t-quantile and SE are shared; only the inflation differs).
    assert nb_halfwidth == pytest.approx(naive_halfwidth * factor, rel=1e-12)
    # And it is a meaningful (>=30%) widening, not noise -- catches a silent revert to no-correction.
    assert nb_halfwidth >= naive_halfwidth * 1.30


def test_auto_inflation_mean_minus_std_with_geometry() -> None:
    # The geometry path also drives mean_minus_std (it multiplies the std spread, not the SE).
    """Auto inflation mean minus std with geometry."""
    scores = [1.0, 1.2, 0.9, 1.1, 1.05]
    k = len(scores)
    naive = aggregate_fold_scores(scores, mode="mean_minus_std", direction="min", alpha=1.0, correlation_inflation=1.0)
    nb = aggregate_fold_scores(scores, mode="mean_minus_std", direction="min", alpha=1.0, split_geometry=(k, 1.0 / k))
    mean = float(np.mean(scores))
    factor = nadeau_bengio_inflation(k, 1.0 / k)
    assert (nb - mean) == pytest.approx((naive - mean) * factor, rel=1e-12)


def test_explicit_float_overrides_geometry() -> None:
    # An explicit numeric factor is applied verbatim and IGNORES split_geometry (caller knows best).
    """Explicit float overrides geometry."""
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    explicit = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, correlation_inflation=2.0, split_geometry=(5, 0.2))
    expected = aggregate_fold_scores(scores, mode="t_lcb", direction="min", confidence=0.9, correlation_inflation=2.0)
    assert explicit == pytest.approx(expected, rel=1e-12)


def test_bad_inflation_string_raises() -> None:
    """Bad inflation string raises."""
    with pytest.raises(ValueError, match="must be a float"):
        aggregate_fold_scores([1.0, 2.0, 3.0], mode="t_lcb", correlation_inflation="bogus")  # type: ignore[arg-type]
