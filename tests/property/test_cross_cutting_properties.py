"""Cross-cutting property / regression tests spanning metrics, calibration, core and utils.

These pin invariants that every affected public API must satisfy regardless of the
concrete inputs, plus a handful of exact-contract sensors (silent-sentinel returns,
pickle round-trips, NaN-safe argmax). They complement the per-module tests by asserting
the SHARED guarantees the audit's cross-cutting list calls out:

- probabilistic metrics (Brier, empirical coverage) stay in [0, 1];
- pinball loss is symmetric in (y, q) at alpha=0.5 and equals half the MAE there;
- CRPS of a constant (point) quantile forecast equals mean(|forecast - obs|) in closed form;
- frame-compat conversion preserves object identity for pandas/ndarray and dtypes/shape for polars;
- public estimator/config classes survive a pickle round-trip and re-predict identically;
- argmax over NaN-bearing scores never silently returns the NaN slot;
- documented sentinel returns (Brier/log-loss NaN on bad probs, quantile_safe all-NaN fallback) are exact.

``hypothesis`` is available on this stack, so the pure invariants use ``@given``; the
closed-form / contract sensors use plain parametrization over hand-picked edge inputs.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from mlframe.metrics.quantile import (
    coverage,
    crps_from_quantiles,
    mean_interval_width,
    pinball_loss,
    pinball_loss_per_alpha,
    winkler_score,
)
from mlframe.metrics._core_auc_brier import fast_brier_score_loss, brier_score_loss
from mlframe.metrics._log_loss_and_separation import fast_log_loss
from mlframe.utils.nan_safe import argmax_classes_safe, quantile_safe, median_safe
from mlframe.core.frame_compat import to_pandas_or_array
from mlframe.calibration.post import BinaryPostCalibrator
from mlframe.calibration.policy import CalibrationConfig
from sklearn.isotonic import IsotonicRegression


# Warm the numba kernels once at import so the first ``@given`` example does not pay the JIT
# compile inside a timed hypothesis example. Deadline is disabled on the property tests anyway,
# but this keeps the whole file's wall-time predictable. Failures here are non-fatal.
try:  # pragma: no cover - warmup only
    _wy = np.array([0, 1, 0, 1], dtype=np.int64)
    _wp = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)
    fast_brier_score_loss(_wy, _wp)
    fast_log_loss(_wy, _wp)
    pinball_loss(_wy.astype(np.float64), _wp, 0.5)
    pinball_loss_per_alpha(_wy.astype(np.float64), np.column_stack([_wp, _wp]), [0.3, 0.7])
    coverage(_wp, _wp - 0.1, _wp + 0.1)
    winkler_score(_wp, _wp - 0.1, _wp + 0.1, 0.2)
    crps_from_quantiles(_wy.astype(np.float64), np.column_stack([_wp, _wp]), [0.3, 0.7])
except Exception:  # pragma: no cover
    pass


_PROP_SETTINGS = settings(deadline=None, max_examples=60)


# --------------------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------------------


@st.composite
def _binary_y_and_probs(draw, min_n: int = 1, max_n: int = 40):
    """Matched-length (y in {0,1}, p in [0,1]) pair with finite entries."""
    n = draw(st.integers(min_n, max_n))
    y = draw(arrays(np.int64, n, elements=st.integers(0, 1)))
    p = draw(arrays(np.float64, n, elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)))
    return y, p


@st.composite
def _y_lo_width(draw, min_n: int = 1, max_n: int = 40):
    """(y, q_lo, q_hi) with q_hi >= q_lo (a well-formed interval) and bounded magnitude."""
    n = draw(st.integers(min_n, max_n))
    elem = st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False)
    y = draw(arrays(np.float64, n, elements=elem))
    lo = draw(arrays(np.float64, n, elements=elem))
    width = draw(arrays(np.float64, n, elements=st.floats(0.0, 500.0, allow_nan=False, allow_infinity=False)))
    hi = lo + width
    return y, lo, hi


@st.composite
def _y_and_quantile_preds(draw, min_n: int = 1, max_n: int = 20):
    """(y, preds_NK, alphas) with K>=1 aligned columns; preds arbitrary finite floats."""
    n = draw(st.integers(min_n, max_n))
    k = draw(st.integers(1, 5))
    elem = st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False)
    y = draw(arrays(np.float64, n, elements=elem))
    preds = draw(arrays(np.float64, (n, k), elements=elem))
    alphas = list(np.linspace(0.1, 0.9, k))
    return y, preds, alphas


# --------------------------------------------------------------------------------------------------
# Property invariants: probabilistic metrics stay in [0, 1]
# --------------------------------------------------------------------------------------------------


@_PROP_SETTINGS
@given(data=_binary_y_and_probs())
def test_brier_score_stays_in_unit_interval(data):
    """Brier score = mean((y - p)^2) is bounded to [0, 1] for y in {0,1}, p in [0,1]."""
    y, p = data
    b = fast_brier_score_loss(y, p)
    assert np.isfinite(b)
    assert -1e-12 <= b <= 1.0 + 1e-9, f"brier {b} outside [0,1]"
    # Documented alias must be the same callable / same value.
    assert brier_score_loss(y, p) == b


@_PROP_SETTINGS
@given(data=_y_lo_width(), alpha_miscov=st.floats(0.02, 0.98, allow_nan=False))
def test_coverage_in_unit_interval_and_winkler_geq_width(data, alpha_miscov):
    """Empirical coverage is a fraction in [0,1]; Winkler score >= mean interval width
    because the miscoverage penalty is pointwise non-negative."""
    y, lo, hi = data
    cov = coverage(y, lo, hi)
    assert 0.0 <= cov <= 1.0, f"coverage {cov} outside [0,1]"
    w = winkler_score(y, lo, hi, alpha_miscov)
    width = mean_interval_width(lo, hi)
    assert np.isfinite(w) and np.isfinite(width)
    assert w >= width - 1e-6, f"winkler {w} < mean_width {width}"


# --------------------------------------------------------------------------------------------------
# Property invariants: pinball loss
# --------------------------------------------------------------------------------------------------


@_PROP_SETTINGS
@given(data=_y_and_quantile_preds())
def test_pinball_symmetry_and_half_mae_at_alpha_half(data):
    """At alpha=0.5 pinball loss is symmetric in (y, q) and equals 0.5 * mean(|y - q|)."""
    y, preds, _ = data
    q = preds[:, 0]
    forward = pinball_loss(y, q, 0.5)
    swapped = pinball_loss(q, y, 0.5)
    half_mae = 0.5 * float(np.mean(np.abs(y - q)))
    assert forward == pytest.approx(swapped, abs=1e-9), "alpha=0.5 pinball not symmetric in (y,q)"
    assert forward == pytest.approx(half_mae, abs=1e-9), "alpha=0.5 pinball != 0.5*MAE"


@_PROP_SETTINGS
@given(data=_y_and_quantile_preds(), alpha=st.floats(0.01, 0.99, allow_nan=False))
def test_pinball_nonneg_and_per_alpha_matches_scalar(data, alpha):
    """Pinball loss is non-negative for any alpha in (0,1), and the fused per-alpha API
    agrees column-for-column with the scalar wrapper."""
    y, preds, alphas = data
    assert pinball_loss(y, preds[:, 0], alpha) >= -1e-12
    per = pinball_loss_per_alpha(y, preds, alphas)
    for j, a in enumerate(alphas):
        assert per[float(a)] == pytest.approx(pinball_loss(y, preds[:, j], a), abs=1e-9)


# --------------------------------------------------------------------------------------------------
# CRPS closed form on a known case
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alphas", [[0.1, 0.9], [0.05, 0.25, 0.5, 0.75, 0.95], [0.2, 0.4, 0.6, 0.8]])
@pytest.mark.parametrize("c", [1.5, -3.0, 0.0])
def test_crps_point_forecast_equals_mean_abs_error(alphas, c):
    """CRPS(F, y) with a degenerate (constant) predicted quantile function q(alpha)=c equals
    mean(|c - y|). The pinball loss is exactly linear in alpha for fixed q, so the trapezoidal
    tail-extended integral is exact -- this pins the tail-integral handling, not just the interior."""
    y = np.array([2.0, -1.0, 5.0, c], dtype=np.float64)
    preds = np.full((y.shape[0], len(alphas)), c, dtype=np.float64)
    got = crps_from_quantiles(y, preds, alphas)
    expected = float(np.mean(np.abs(y - c)))
    assert got == pytest.approx(expected, abs=1e-9), f"CRPS {got} != mean|y-c| {expected}"


def test_crps_denser_grid_agrees_with_coarse_on_point_forecast():
    """For the exact point-forecast case, refining the alpha grid must not change CRPS."""
    y = np.array([3.0, -2.0, 0.5], dtype=np.float64)
    c = 1.0
    coarse = crps_from_quantiles(y, np.full((3, 2), c), [0.25, 0.75])
    dense = crps_from_quantiles(y, np.full((3, 9), c), list(np.linspace(0.1, 0.9, 9)))
    assert coarse == pytest.approx(dense, abs=1e-9)


# --------------------------------------------------------------------------------------------------
# argmax NaN-safety
# --------------------------------------------------------------------------------------------------


def test_argmax_classes_safe_does_not_pick_nan_slot():
    """A row whose numerical max slot is NaN must resolve to the finite argmax, never the NaN index.
    Naive np.argmax returns the NaN position (verified), so this pins the safe behaviour."""
    probs = np.array(
        [
            [0.2, np.nan, 0.5],   # naive argmax -> 1 (nan); safe -> 2
            [np.nan, 0.7, 0.1],   # safe -> 1
            [0.9, 0.05, 0.05],    # all finite -> 0
        ]
    )
    # Confirm the bug the helper guards against actually exists in raw numpy.
    assert np.argmax(probs[0]) == 1
    out = argmax_classes_safe(probs, context="test")
    assert out.tolist() == [2, 1, 0]
    assert out.dtype == np.int64


def test_argmax_classes_safe_all_nan_row_uses_fallback_class():
    """A fully non-finite row gets the explicit fallback_class, not an arbitrary NaN slot."""
    probs = np.array([[np.nan, np.nan], [0.3, 0.8]])
    out = argmax_classes_safe(probs, fallback_class=7, context="test")
    assert out.tolist() == [7, 1]


# --------------------------------------------------------------------------------------------------
# Silent-sentinel contract sensors
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_prob",
    [
        np.array([0.1, 1.5, 0.3]),     # > 1
        np.array([-0.2, 0.4, 0.6]),    # < 0
        np.array([0.1, np.nan, 0.3]),  # NaN
    ],
)
def test_brier_returns_exact_nan_on_bad_probs(bad_prob):
    """fast_brier_score_loss pins its documented sentinel: exactly NaN on out-of-[0,1] / NaN probs,
    rather than squaring garbage into a plausible-looking score."""
    y = np.array([0, 1, 1], dtype=np.int64)
    val = fast_brier_score_loss(y, bad_prob)
    assert isinstance(val, float) and np.isnan(val)


def test_log_loss_returns_exact_nan_on_bad_probs_and_single_class():
    """fast_log_loss returns exactly NaN on out-of-range probs and on single-class y."""
    y = np.array([0, 1, 1], dtype=np.int64)
    assert np.isnan(fast_log_loss(y, np.array([0.1, 1.2, 0.3])))     # prob > 1
    assert np.isnan(fast_log_loss(y, np.array([0.1, -0.1, 0.3])))    # prob < 0
    # Single-class y (all zeros) with valid probs -> NaN (mirrors sklearn's undefined case).
    assert np.isnan(fast_log_loss(np.array([0, 0, 0]), np.array([0.2, 0.4, 0.6])))


@pytest.mark.parametrize("sentinel", [float("nan"), -1.0, -999.0])
def test_quantile_safe_all_nan_returns_exact_sentinel(sentinel):
    """quantile_safe returns EXACTLY the requested fallback on an all-NaN input (scalar q)."""
    arr = np.array([np.nan, np.nan, np.nan])
    out = quantile_safe(arr, 0.5, fallback=sentinel)
    if np.isnan(sentinel):
        assert np.isnan(out)
    else:
        assert out == sentinel


def test_quantile_safe_sequence_q_all_nan_returns_sentinel_vector():
    """With a sequence q, the all-NaN fallback keeps nanquantile's array output shape."""
    arr = np.array([np.nan, np.nan])
    out = quantile_safe(arr, [0.25, 0.5, 0.75], fallback=-1.0)
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    assert np.all(out == -1.0)


def test_median_safe_all_nan_returns_exact_sentinel():
    assert median_safe(np.array([np.nan, np.nan]), fallback=-5.0) == -5.0
    # A finite value present -> real median over finite entries (NaN ignored).
    assert median_safe(np.array([np.nan, 2.0, 4.0])) == pytest.approx(3.0)


# --------------------------------------------------------------------------------------------------
# Frame-compat: dtype / shape / identity preservation
# --------------------------------------------------------------------------------------------------


def test_frame_compat_pandas_and_ndarray_return_same_object():
    """pandas DataFrame / Series and numpy ndarray pass through untouched (no copy, same identity)."""
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    s = pd.Series([1, 2, 3])
    arr = np.arange(6).reshape(2, 3)
    assert to_pandas_or_array(df) is df
    assert to_pandas_or_array(s) is s
    assert to_pandas_or_array(arr) is arr


def test_frame_compat_polars_dataframe_preserves_shape_and_dtypes():
    """Polars DataFrame -> pandas without collapsing numeric dtypes to object (the prod bug this
    helper exists to prevent: Float32 columns being mis-read as high-cardinality categoricals)."""
    pf = pl.DataFrame(
        {
            "f32": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
            "i64": pl.Series([10, 20, 30], dtype=pl.Int64),
        }
    )
    out = to_pandas_or_array(pf)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (3, 2)
    assert out["f32"].dtype == np.float32
    assert out["i64"].dtype == np.int64
    assert out.columns.tolist() == ["f32", "i64"]


def test_frame_compat_polars_series_preserves_length_and_dtype():
    ps = pl.Series("s", [1.0, 2.0, 3.0, 4.0], dtype=pl.Float32)
    out = to_pandas_or_array(ps)
    assert isinstance(out, pd.Series)
    assert len(out) == 4
    assert out.dtype == np.float32


# --------------------------------------------------------------------------------------------------
# Pickle round-trips for public classes
# --------------------------------------------------------------------------------------------------


def test_pickle_binary_post_calibrator_repredicts_identically():
    """A fitted BinaryPostCalibrator survives pickle and re-predicts byte-identical calibrated probs.
    Uses a tiny sklearn IsotonicRegression as the wrapped calibrator so the fit is cheap."""
    rng = np.random.default_rng(0)
    calib_p = np.linspace(0.01, 0.99, 40)
    calib_y = (calib_p + rng.normal(0, 0.05, size=40) > 0.5).astype(np.int64)
    cal = BinaryPostCalibrator(
        calibrator=IsotonicRegression(out_of_bounds="clip"),
        fit_method_name="fit",
        transform_method_name="transform",
    )
    cal.fit(calib_p, calib_y)
    test_p = np.linspace(0.05, 0.95, 15)
    before = cal.predict_proba(test_p)
    restored = pickle.loads(pickle.dumps(cal))
    after = restored.predict_proba(test_p)
    assert before.shape == after.shape == (15, 2)
    np.testing.assert_array_equal(before, after)
    # Fitted sklearn bookkeeping must also survive.
    assert np.array_equal(restored.classes_, cal.classes_)
    assert restored.n_features_in_ == cal.n_features_in_


def test_pickle_calibration_config_preserves_all_fields():
    """CalibrationConfig (public dataclass) round-trips with every field intact."""
    cfg = CalibrationConfig(
        policy_auto_pick=False,
        emit_plot=True,
        plot_path="reports/x.png",
        n_bootstrap=250,
        alpha=0.1,
        candidates=("Isotonic", "Beta"),
        selection="oof",
        inner_cv_splits=3,
    )
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored == cfg
    assert restored.candidates == ("Isotonic", "Beta")
    assert restored.n_bootstrap == 250
    assert restored.selection == "oof"
