"""biz_value tests for ``training.OverlappingWalkForwardCV`` and ``training.cv_stability_check``.

Win 1: overlapping-window walk-forward CV (more, correlated folds) gives a lower-variance estimate of the
mean CV metric across repeated draws of the underlying data than non-overlapping walk-forward CV (fewer,
independent folds) at the same window/gap/test configuration -- the writeup's stated motivation for using it.
Win 2: ``cv_stability_check`` correctly separates a smooth, cross-seed-consistent hyperparameter curve from a
jagged, seed-inconsistent one -- i.e. it should actually catch noise-chasing before a real deployment does.
"""

from __future__ import annotations

import numpy as np

from mlframe.training import OverlappingWalkForwardCV, cv_stability_check


def _mean_abs_error_cv(splitter, y: np.ndarray) -> float:
    errors = []
    for train_idx, test_idx in splitter.split(y):
        pred = float(np.mean(y[train_idx]))
        errors.extend(np.abs(y[test_idx] - pred))
    return float(np.mean(errors)) if errors else float("nan")


def test_biz_val_overlapping_walk_forward_cv_lowers_estimate_variance():
    n_samples = 260
    window_length, gap, test_length = 40, 5, 5
    overlapping = OverlappingWalkForwardCV(window_length=window_length, step=10, gap=gap, test_length=test_length)
    non_overlapping = OverlappingWalkForwardCV(window_length=window_length, step=window_length + gap + test_length, gap=gap, test_length=test_length)

    assert overlapping.get_n_splits(X=np.zeros(n_samples)) > non_overlapping.get_n_splits(X=np.zeros(n_samples))

    np.random.default_rng(0)
    overlap_estimates = []
    non_overlap_estimates = []
    for trial in range(60):
        trial_rng = np.random.default_rng(1000 + trial)
        y = 5.0 + 0.5 * np.sin(np.arange(n_samples) / 15.0) + trial_rng.normal(0, 1.0, size=n_samples)
        overlap_estimates.append(_mean_abs_error_cv(overlapping, y))
        non_overlap_estimates.append(_mean_abs_error_cv(non_overlapping, y))

    overlap_std = float(np.std(overlap_estimates, ddof=1))
    non_overlap_std = float(np.std(non_overlap_estimates, ddof=1))

    assert overlap_std < non_overlap_std, (
        f"overlapping-window CV should give a lower-variance mean-metric estimate: overlap_std={overlap_std:.4f} non_overlap_std={non_overlap_std:.4f}"
    )


def test_overlapping_walk_forward_cv_folds_respect_gap_and_length():
    y = np.arange(100)
    splitter = OverlappingWalkForwardCV(window_length=20, step=10, gap=3, test_length=5)
    folds = list(splitter.split(y))
    assert len(folds) == splitter.get_n_splits(X=y)
    for train_idx, test_idx in folds:
        assert len(train_idx) == 20
        assert len(test_idx) == 5
        assert test_idx[0] - train_idx[-1] == 3 + 1  # gap samples strictly between train end and test start


def test_biz_val_cv_stability_check_separates_smooth_from_jagged_curves():
    rng = np.random.default_rng(0)
    hp_grid = np.linspace(0, 1, 15)

    smooth_curves = [-((hp_grid - 0.6) ** 2) + rng.normal(0, 0.01, size=len(hp_grid)) for _ in range(5)]
    smooth_result = cv_stability_check(smooth_curves)

    jagged_curves = [rng.normal(0, 1.0, size=len(hp_grid)) for _ in range(5)]
    jagged_result = cv_stability_check(jagged_curves)

    assert smooth_result["stable"] is True, smooth_result
    assert jagged_result["stable"] is False, jagged_result
    assert smooth_result["cross_seed_argmax_agreement"] > jagged_result["cross_seed_argmax_agreement"]


def test_cv_stability_check_too_few_seeds_raises():
    import pytest

    with pytest.raises(ValueError):
        cv_stability_check([[0.1, 0.2, 0.3]], min_seeds=2)


def _persistence_mae_cv(splitter: OverlappingWalkForwardCV, y: np.ndarray) -> float:
    """Last-observed-value predictor: directly exploits whatever autocorrelation survives the fold's gap."""
    errors = []
    for train_idx, test_idx in splitter.split(y, y=y):
        pred = y[train_idx[-1]]
        errors.extend(np.abs(y[test_idx] - pred))
    return float(np.mean(errors)) if errors else float("nan")


def test_biz_val_overlapping_walk_forward_cv_adaptive_gap_avoids_leakage_inflated_score():
    """Adaptive-gap mode should self-widen the embargo on a strongly-autocorrelated label and land close to the
    TRUE held-out-future score, while a caller-picked small fixed gap silently produces a leakage-inflated
    (too-optimistic) CV score on the same data -- the whole point of not requiring hand-tuned ``gap``.
    """
    rng = np.random.default_rng(0)
    n_samples, phi = 2_000, 0.93
    y = np.zeros(n_samples)
    for i in range(1, n_samples):
        y[i] = phi * y[i - 1] + rng.normal(0, 1)

    window_length, step, test_length, small_gap = 200, 40, 10, 1
    fixed_splitter = OverlappingWalkForwardCV(window_length=window_length, step=step, gap=small_gap, test_length=test_length)
    adaptive_splitter = OverlappingWalkForwardCV(
        window_length=window_length,
        step=step,
        gap=small_gap,
        test_length=test_length,
        adaptive_gap=True,
        autocorr_threshold=0.2,
    )

    fixed_gap_score = _persistence_mae_cv(fixed_splitter, y)
    adaptive_gap_score = _persistence_mae_cv(adaptive_splitter, y)

    # True held-out-future score: persistence MAE at a gap (40) beyond this process's decorrelation length, swept
    # across the same fold starts as the splitters above -- the honest generalization number neither CV score sees.
    true_gap = 40
    true_errors = []
    for train_start in range(0, n_samples - window_length - true_gap - test_length, step):
        train_end = train_start + window_length
        pred = y[train_end - 1]
        test_start = train_end + true_gap
        test_end = test_start + test_length
        if test_end > n_samples:
            break
        true_errors.extend(np.abs(y[test_start:test_end] - pred))
    true_future_score = float(np.mean(true_errors))

    fixed_gap_error_vs_truth = abs(fixed_gap_score - true_future_score)
    adaptive_gap_error_vs_truth = abs(adaptive_gap_score - true_future_score)

    assert fixed_gap_score < true_future_score, (
        f"small fixed gap should look leakage-inflated (optimistic) vs the true future score: "
        f"fixed_gap_score={fixed_gap_score:.4f} true_future_score={true_future_score:.4f}"
    )
    assert adaptive_gap_error_vs_truth < 0.65 * fixed_gap_error_vs_truth, (
        f"adaptive gap should land materially closer to the true future score than a hand-picked small fixed gap: "
        f"fixed_gap_score={fixed_gap_score:.4f} adaptive_gap_score={adaptive_gap_score:.4f} "
        f"true_future_score={true_future_score:.4f} fixed_err={fixed_gap_error_vs_truth:.4f} adaptive_err={adaptive_gap_error_vs_truth:.4f}"
    )


def test_overlapping_walk_forward_cv_adaptive_gap_off_by_default_is_bit_identical():
    """Omitting the new adaptive-gap params must reproduce the exact pre-existing fixed-gap fold indices."""
    y = np.arange(300, dtype=np.float64)
    explicit_default = OverlappingWalkForwardCV(window_length=50, step=15, gap=4, test_length=6, adaptive_gap=False)
    implicit_default = OverlappingWalkForwardCV(window_length=50, step=15, gap=4, test_length=6)

    folds_a = list(explicit_default.split(y, y=y))
    folds_b = list(implicit_default.split(y))

    assert len(folds_a) == len(folds_b) == explicit_default.get_n_splits(X=y) == implicit_default.get_n_splits(X=y)
    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)
