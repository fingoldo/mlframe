"""biz_value test for mlframe.competition.trend_noise_decorrelation.inject_noise_and_recenter.

COMPETITION / EXPLORATORY ONLY — see the module docstring for why this trick
must never be used in production.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.competition.trend_noise_decorrelation import inject_noise_and_recenter


def _make_segments(
    n_segments: int, seg_len: int, base_noise_std: float, trend_scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train/test-style segments sharing structure but differing by a spurious linear trend in "test"."""
    segments = np.empty((2 * n_segments, seg_len), dtype=np.float64)
    is_test = np.zeros(2 * n_segments, dtype=np.int64)
    peak_target = np.empty(2 * n_segments, dtype=np.float64)

    for i in range(2 * n_segments):
        base = rng.normal(0.0, base_noise_std, size=seg_len)
        n_peaks = rng.integers(1, 4)
        peak_positions = rng.choice(seg_len, size=n_peaks, replace=False)
        peak_magnitude = float(rng.uniform(3.0, 8.0))
        base[peak_positions] += peak_magnitude
        peak_target[i] = peak_magnitude

        group = i >= n_segments
        is_test[i] = int(group)
        if group:
            trend = np.linspace(0.0, trend_scale, seg_len)
            base = base + trend

        segments[i] = base

    return segments, is_test, peak_target


def _adversarial_auc(segments: np.ndarray, is_test: np.ndarray) -> float:
    """Adversarial-validation AUC from mean/median location features.

    Deliberately restricted to location features (mean/median) rather than
    spread/peak features (std/min/max) because that is exactly what the real
    trick targets: it destroys the non-stationary trend contaminating
    mean/quantile-based location features while leaving peak/volatility
    features (dominated by the injected peaks) largely untouched. A linear
    model is used (not a deep ensemble) so near-chance separability reads out
    as AUC close to 0.5 rather than small-sample tree-overfit noise.
    """
    feats = np.column_stack(
        [
            segments.mean(axis=1),
            np.median(segments, axis=1),
        ]
    )
    clf = LogisticRegression()
    scores = cross_val_score(clf, feats, is_test, cv=5, scoring="roc_auc")
    return float(np.mean(scores))


def test_biz_val_trend_noise_decorrelation_defeats_adversarial_trend_and_preserves_peak_signal() -> None:
    """Noise+recenter drops adversarial train/test AUC from >=0.90 toward chance while preserving peak-magnitude correlation."""
    rng = np.random.default_rng(42)
    n_segments = 150
    seg_len = 400
    base_noise_std = 0.4
    trend_scale = 6.0

    segments, is_test, peak_target = _make_segments(n_segments, seg_len, base_noise_std, trend_scale, rng)

    raw_auc = _adversarial_auc(segments, is_test)
    # the injected linear trend must be genuinely adversarially detectable pre-treatment
    assert raw_auc >= 0.90

    treated = np.stack([inject_noise_and_recenter(seg, noise_std=0.5, random_state=123) for seg in segments])

    treated_auc = _adversarial_auc(treated, is_test)
    # noise+recenter must measurably reduce adversarial-detectability toward chance
    assert treated_auc <= 0.65
    assert treated_auc < raw_auc - 0.25

    # downstream peak/volatility signal must be preserved: max-abs-value still tracks peak magnitude
    treated_peak_proxy = np.abs(treated).max(axis=1)
    preserved_corr = float(np.corrcoef(treated_peak_proxy, peak_target)[0, 1])
    assert preserved_corr >= 0.65

    raw_peak_proxy = np.abs(segments).max(axis=1)
    raw_corr_peak = float(np.corrcoef(raw_peak_proxy, peak_target)[0, 1])
    # peak/target correlation must not collapse relative to the untreated signal
    assert preserved_corr >= raw_corr_peak - 0.10


def test_biz_val_trend_noise_decorrelation_fixed_seed_reproducible() -> None:
    """Same random_state gives bit-identical output; a different random_state gives a different result."""
    rng = np.random.default_rng(0)
    segment = rng.normal(0.0, 1.0, size=200)

    out_a = inject_noise_and_recenter(segment, noise_std=0.5, random_state=7)
    out_b = inject_noise_and_recenter(segment, noise_std=0.5, random_state=7)
    np.testing.assert_array_equal(out_a, out_b)

    out_c = inject_noise_and_recenter(segment, noise_std=0.5, random_state=8)
    assert not np.array_equal(out_a, out_c)


def test_biz_val_trend_noise_decorrelation_recenters_to_zero_median() -> None:
    """Output is recentered so its median is (near-)exactly zero regardless of the input's original center."""
    rng = np.random.default_rng(1)
    segment = rng.normal(5.0, 2.0, size=1000)
    out = inject_noise_and_recenter(segment, noise_std=0.5, random_state=1)
    assert abs(float(np.median(out))) < 1e-9


def test_biz_val_trend_noise_decorrelation_rejects_empty_or_2d() -> None:
    """Empty input, 2D input, or a negative noise_std each raise ValueError."""
    with pytest.raises(ValueError):
        inject_noise_and_recenter(np.array([]), noise_std=0.5)
    with pytest.raises(ValueError):
        inject_noise_and_recenter(np.zeros((2, 2)), noise_std=0.5)
    with pytest.raises(ValueError):
        inject_noise_and_recenter(np.zeros(10), noise_std=-1.0)
