"""Unit tests for the outlier-robust basis-axis gate and preprocessors.

Covers the spike-contamination detector (``_detect_heavy_tail``), the robust bounds helper (``_robust_lo_hi``), the gated
preprocessors (``_preprocess_zscore`` / ``_preprocess_minmax_neg1_1`` / ``_preprocess_shift_nonneg``), their apply
counterparts, and the Fourier axis (``_fit_fourier_for_col``). The central contract: clean / naturally-heavy-tailed
columns take the byte-identical LEGACY path; only spike-CONTAMINATED columns take the robust path.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import (
    _apply_minmax,
    _apply_shift,
    _apply_zscore,
    _detect_heavy_tail,
    _preprocess_minmax_neg1_1,
    _preprocess_shift_nonneg,
    _preprocess_zscore,
    _robust_axis_enabled,
    _robust_lo_hi,
)
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fit_fourier_for_col


def _spike(rng, n=3000, frac=0.05, scale=1000.0):
    x = rng.standard_normal(n)
    idx = rng.choice(n, max(1, int(n * frac)), replace=False)
    x[idx] = rng.choice([-1.0, 1.0], idx.size) * scale
    return x


# ---------------------------------------------------------------------------
# Detector: clean / natural-heavy -> False; spike contamination -> True.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    "gen",
    [
        pytest.param(lambda r: r.standard_normal(3000), id="gauss"),
        pytest.param(lambda r: r.uniform(-1.0, 1.0, 3000), id="uniform"),
        pytest.param(lambda r: r.lognormal(0.0, 1.2, 3000), id="lognormal"),
        pytest.param(lambda r: r.lognormal(size=3000) - r.lognormal(size=3000), id="lognorm_diff"),
        pytest.param(lambda r: r.standard_t(3, 3000), id="student_t3"),
        pytest.param(lambda r: r.exponential(1.0, 3000), id="exponential"),
        pytest.param(lambda r: r.gamma(2.0, 1.0, 3000), id="gamma"),
        pytest.param(lambda r: np.clip(r.standard_t(3, 3000), -8.0, 8.0), id="clipped_t3"),
    ],
)
def test_detector_clean_and_natural_heavy_not_flagged(seed, gen):
    """A clean Gaussian/uniform OR a genuinely heavy-tailed-but-clean column (lognormal, t3, exponential, gamma) whose tail
    is CONTINUOUS with the bulk must NOT trip the gate -- robustifying these would change byte values on legitimate data."""
    x = gen(np.random.default_rng(seed))
    assert _detect_heavy_tail(x) is False


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("frac", [0.01, 0.05])
@pytest.mark.parametrize("scale", [50.0, 100.0, 1000.0])
def test_detector_spike_contamination_flagged(seed, frac, scale):
    """A thin fraction of order-of-magnitude spikes separated from the bulk by a multiplicative gap MUST trip the gate
    across contamination fractions (1-5%) and scales (50x-1000x)."""
    x = _spike(np.random.default_rng(seed), frac=frac, scale=scale)
    assert _detect_heavy_tail(x) is True


def test_detector_degenerate_columns_never_trip():
    """Constant, near-constant, too-short, and all-non-finite columns have no scale to corrupt -> gate stays off."""
    assert _detect_heavy_tail(np.full(3000, 5.0)) is False
    assert _detect_heavy_tail(np.array([1.0, 2.0, 3.0])) is False  # < 8 finite values
    assert _detect_heavy_tail(np.full(3000, np.nan)) is False
    near_const = np.full(3000, 2.0)
    near_const[0] = 2.0 + 1e-15
    assert _detect_heavy_tail(near_const) is False


def test_detector_ignores_nans_in_finite_subset():
    """A spike-contaminated column with interspersed NaNs is still detected (NaNs are filtered before the gap test)."""
    rng = np.random.default_rng(0)
    x = _spike(rng, frac=0.05, scale=1000.0)
    x[rng.choice(x.size, 200, replace=False)] = np.nan
    assert _detect_heavy_tail(x) is True


# ---------------------------------------------------------------------------
# Robust bounds: MAD-anchored, contamination-proof.
# ---------------------------------------------------------------------------


def test_robust_lo_hi_anchored_to_clean_core_not_outliers():
    """``_robust_lo_hi`` must track the CLEAN core (~+/-3 sigma) regardless of how extreme the injected spikes are -- a MAD
    anchor ignores up to ~50% contamination, unlike a raw min/max or a shallow inner-quantile trim."""
    rng = np.random.default_rng(0)
    x = _spike(rng, frac=0.05, scale=1000.0)
    lo, hi = _robust_lo_hi(x)
    # Clean N(0,1) core => bounds near +/-3, NOT anywhere near the +/-1000 spikes.
    assert -6.0 < lo < -1.5 and 1.5 < hi < 6.0, f"robust bounds leaked into the spike mass: ({lo}, {hi})"


# ---------------------------------------------------------------------------
# Gated preprocessors: clean -> legacy params (no clip); outlier -> robust (clip).
# ---------------------------------------------------------------------------


def test_zscore_clean_takes_legacy_path():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(3000)
    z, params = _preprocess_zscore(x)
    assert "clip" not in params
    assert params["mean"] == pytest.approx(float(np.mean(x)))
    assert params["std"] == pytest.approx(float(np.std(x) + 1e-12))


def test_zscore_outlier_takes_robust_path_and_clamps():
    rng = np.random.default_rng(1)
    x = _spike(rng, frac=0.05, scale=1000.0)
    z, params = _preprocess_zscore(x)
    assert "clip" in params
    # Robust std ~ 1 (the clean-core sigma), NOT the ~150 raw std the legacy path would compute.
    assert 0.5 < params["std"] < 2.0
    # Mapped axis is clamped to +/-clip -- a 1000x spike lands at the working-domain edge, not at z ~ 1000.
    assert float(np.max(np.abs(z))) <= params["clip"] + 1e-9


def test_minmax_outlier_clamps_to_unit_interval():
    rng = np.random.default_rng(2)
    x = _spike(rng, frac=0.05, scale=1000.0)
    z, params = _preprocess_minmax_neg1_1(x)
    assert params.get("clip") == 1.0
    assert float(z.min()) >= -1.0 - 1e-9 and float(z.max()) <= 1.0 + 1e-9


def test_shift_outlier_clamps_upper_tail():
    rng = np.random.default_rng(3)
    x = np.abs(_spike(rng, frac=0.05, scale=1000.0)) + 0.01  # positive domain for the Laguerre shift.
    z, params = _preprocess_shift_nonneg(x)
    assert "clip" in params
    assert float(z.min()) >= 0.0
    assert float(z.max()) <= params["clip"] + 1e-9


# ---------------------------------------------------------------------------
# Apply path replays the stored clip consistently (fit/transform agreement).
# ---------------------------------------------------------------------------


def test_apply_replays_clip_consistently():
    """The apply functions must reproduce the fit-time clamp from the stored params so a replayed recipe maps a held-out
    extreme row to the same clamped edge value as at fit time."""
    rng = np.random.default_rng(4)
    x = _spike(rng, frac=0.05, scale=1000.0)
    z_fit, p = _preprocess_zscore(x)
    np.testing.assert_array_equal(_apply_zscore(x, p), z_fit)

    z_fit_m, pm = _preprocess_minmax_neg1_1(x)
    np.testing.assert_array_equal(_apply_minmax(x, pm), z_fit_m)

    xp = np.abs(x) + 0.01
    z_fit_s, ps = _preprocess_shift_nonneg(xp)
    np.testing.assert_array_equal(_apply_shift(xp, ps), z_fit_s)


def test_apply_legacy_params_without_clip_unchanged():
    """Legacy params (no ``clip`` key) must apply with NO clamping -- back-compat for recipes fit before the robust path."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(100) * 10.0
    assert np.array_equal(_apply_zscore(x, {"mean": 0.0, "std": 1.0}), x)
    assert np.allclose(_apply_minmax(x, {"lo": -100.0, "hi": 100.0}), 2 * (x + 100.0) / 200.0 - 1)


# ---------------------------------------------------------------------------
# Fourier axis gate.
# ---------------------------------------------------------------------------


def test_fourier_clean_legacy_outlier_robust():
    rng = np.random.default_rng(6)
    clean = rng.standard_normal(3000)
    lo_c, sp_c = _fit_fourier_for_col(clean)
    assert lo_c == float(np.min(clean)) and sp_c == pytest.approx(float(np.max(clean) - np.min(clean)))

    x = _spike(rng, frac=0.05, scale=1000.0)
    lo_o, sp_o = _fit_fourier_for_col(x)
    # Robust span tracks the clean core (~6 sigma), not the ~2000 raw min-max span.
    assert sp_o < 20.0, f"Fourier span did not switch to the robust core estimate: {sp_o}"


def test_fourier_all_nonfinite_returns_default():
    lo, sp = _fit_fourier_for_col(np.full(50, np.nan))
    assert (lo, sp) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Env-var override: MLFRAME_ROBUST_AXIS=0 forces the legacy path.
# ---------------------------------------------------------------------------


def test_env_override_forces_legacy(monkeypatch):
    monkeypatch.setenv("MLFRAME_ROBUST_AXIS", "0")
    assert _robust_axis_enabled() is False
    x = _spike(np.random.default_rng(0), frac=0.05, scale=1000.0)
    z, params = _preprocess_zscore(x)  # detector would trip, but the env var forces legacy.
    assert "clip" not in params
    assert params["std"] == pytest.approx(float(np.std(x) + 1e-12))


def test_env_default_on(monkeypatch):
    monkeypatch.delenv("MLFRAME_ROBUST_AXIS", raising=False)
    assert _robust_axis_enabled() is True
