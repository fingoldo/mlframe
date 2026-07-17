"""Unit + biz_value tests for the cross-target ensemble OUTPUT recalibration.

Covers ``mlframe.training.composite.ensemble._calibration.OutputCalibrator`` /
``fit_output_calibrator`` and the opt-in ``calibrate_output`` wiring on
``CompositeCrossTargetEnsemble``.

Properties pinned:
  * monotone: the fitted map preserves ordering (isotonic / sigmoid / linear).
  * off == identity: with no calibrator attached predict is bit-identical to
    the raw blend (the default state).
  * leakage-free: the calibrator is fit only on caller-supplied OOF preds;
    fitting on OOF then scoring OOF must lower RMSE; the map never inverts
    the ensemble ranking.
  * biz_value: on a synthetic where the raw ensemble blend has a systematic
    S-shaped miscalibration, isotonic recalibration lowers holdout RMSE and
    straightens the calibration curve vs the raw blend.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.ensemble._calibration import (
    OutputCalibrator,
    fit_output_calibrator,
)
from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


class _ConstColModel:
    """Returns a fixed prediction column independent of X (size-tiled)."""

    def __init__(self, col: np.ndarray) -> None:
        self._col = np.asarray(col, dtype=np.float64)

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else self._col.shape[0]
        if self._col.shape[0] == n:
            return self._col
        return np.resize(self._col, n)


def _rmse(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _s_shaped_miscal(y: np.ndarray) -> np.ndarray:
    """Map truth -> an S-shaped (logit-link) miscalibrated raw prediction.

    A monotone but non-affine distortion: compresses the mid-range and
    stretches the tails -- exactly the systematic bias a least-squares blend
    of biased components leaves behind, and exactly what an isotonic map fixes.
    """
    lo, hi = float(y.min()), float(y.max())
    yn = (y - lo) / (hi - lo + 1e-12)
    # squash through tanh around the midpoint, then re-stretch to y-scale.
    distorted = 0.5 + 0.5 * np.tanh(3.0 * (yn - 0.5)) / np.tanh(1.5)
    return lo + (hi - lo) * distorted


# --------------------------------------------------------------------------
# Unit: OutputCalibrator
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["isotonic", "sigmoid", "linear"])
def test_calibrator_is_monotone(method):
    rng = np.random.default_rng(0)
    y = np.sort(rng.uniform(-5, 5, size=500))
    raw = _s_shaped_miscal(y) + rng.normal(0, 0.05, size=y.shape)
    cal = OutputCalibrator(method=method).fit(raw, y)
    grid = np.linspace(raw.min(), raw.max(), 200)
    mapped = cal.predict(grid)
    # Monotone non-decreasing (allow tiny FP slack).
    assert np.all(np.diff(mapped) >= -1e-9), f"{method} map not monotone"


@pytest.mark.parametrize("method", ["isotonic", "sigmoid", "linear"])
def test_calibrator_fit_predict_lowers_rmse(method):
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 10, size=2000)
    raw = _s_shaped_miscal(y) + rng.normal(0, 0.1, size=y.shape)
    cal = OutputCalibrator(method=method).fit(raw, y)
    rmse_raw = _rmse(raw, y)
    rmse_cal = _rmse(cal.predict(raw), y)
    assert rmse_cal <= rmse_raw + 1e-9, f"{method}: calibration must not increase in-fit RMSE (raw={rmse_raw:.4f}, cal={rmse_cal:.4f})"


def test_calibrator_identity_on_degenerate_input():
    # Too few rows -> identity pass-through.
    cal = OutputCalibrator(method="isotonic").fit(np.array([1.0, 2.0]), np.array([5.0, 6.0]))
    got = cal.predict(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(got, [1.0, 2.0, 3.0])
    # Constant predictions -> identity pass-through.
    cal2 = OutputCalibrator(method="sigmoid").fit(np.full(20, 4.0), np.arange(20.0))
    assert np.allclose(cal2.predict(np.array([4.0, 7.0])), [4.0, 7.0])


def test_calibrator_rejects_bad_method():
    with pytest.raises(ValueError):
        OutputCalibrator(method="quadratic")


def test_fit_output_calibrator_returns_none_on_tiny():
    assert fit_output_calibrator(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None
    assert fit_output_calibrator(np.array([]), np.array([])) is None


def test_calibrator_non_finite_passthrough():
    rng = np.random.default_rng(3)
    y = rng.uniform(0, 5, size=300)
    raw = _s_shaped_miscal(y)
    cal = OutputCalibrator(method="isotonic").fit(raw, y)
    out = cal.predict(np.array([np.nan, raw.mean(), np.inf]))
    assert np.isnan(out[0]) and np.isinf(out[2]) and np.isfinite(out[1])


# --------------------------------------------------------------------------
# Unit: ensemble wiring (off == identity, leakage-free)
# --------------------------------------------------------------------------


def _build_ensemble_with_oof(n=2000, seed=7):
    """Two anti-/co-component ensemble whose convex blend is S-miscalibrated."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 10, size=n)
    # Raw blend = miscalibrated S map of y; split it across two components so the
    # equal-weight mean reproduces the miscalibrated surface.
    blend = _s_shaped_miscal(y) + rng.normal(0, 0.1, size=n)
    c0 = blend + rng.normal(0, 0.5, size=n)
    c1 = 2.0 * blend - c0  # mean(c0, c1) == blend exactly
    ens = CompositeCrossTargetEnsemble.from_uniform_weights(
        [_ConstColModel(c0), _ConstColModel(c1)],
        ["c0", "c1"],
    )
    oof_matrix = np.column_stack([c0, c1])
    return ens, oof_matrix, y, blend


def test_off_is_identity_bit_for_bit():
    ens, _oof_matrix, y, blend = _build_ensemble_with_oof()
    X = np.zeros((len(y), 1))
    raw_pred = ens.predict(X)
    assert ens.calibrate_output is False
    assert getattr(ens, "_output_calibrator", None) is None
    # Equal-weight mean of (c0, c1) reproduces blend exactly (bit-for-bit).
    assert np.array_equal(raw_pred, ens.predict(X)), "predict not deterministic"
    assert np.allclose(raw_pred, blend), "off-state blend must equal raw mean"


def test_fit_output_calibrator_attaches_and_lowers_oof_rmse():
    ens, oof_matrix, y, _blend = _build_ensemble_with_oof()
    X = np.zeros((len(y), 1))
    raw_pred = ens.predict(X)
    rmse_raw = _rmse(raw_pred, y)
    ens.fit_output_calibrator(oof_matrix, y, method="isotonic")
    assert ens.calibrate_output is True
    assert ens._output_calibrator is not None
    cal_pred = ens.predict(X)
    rmse_cal = _rmse(cal_pred, y)
    assert rmse_cal < rmse_raw, f"isotonic recalibration must lower OOF RMSE (raw={rmse_raw:.4f}, cal={rmse_cal:.4f})"
    # Ranking preserved: calibration is a monotone post-map of the blend.
    order_raw = np.argsort(raw_pred)
    assert np.all(np.diff(cal_pred[order_raw]) >= -1e-9), "calibration inverted ensemble ranking"


def test_calibrator_survives_pickle():
    import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

    ens, oof_matrix, y, _ = _build_ensemble_with_oof()
    ens.fit_output_calibrator(oof_matrix, y, method="isotonic")
    X = np.zeros((len(y), 1))
    before = ens.predict(X)
    restored = pickle.loads(pickle.dumps(ens))  # nosec B301 -- round-trip of a locally-created, trusted object
    after = restored.predict(X)
    assert np.allclose(before, after), "calibrator did not survive pickle round-trip"
    assert restored.calibrate_output is True


def test_cap_components_drops_stale_calibrator():
    ens, oof_matrix, y, _ = _build_ensemble_with_oof()
    ens.fit_output_calibrator(oof_matrix, y, method="isotonic")
    capped = ens.cap_inference_components(1)
    # Trim changes the blend -> calibrator must be dropped (it would mis-map).
    assert getattr(capped, "_output_calibrator", None) is None
    assert capped.notes.get("output_calibration_dropped_on_cap") is True
    # No-op cap (>= K) keeps the calibrator.
    same = ens.cap_inference_components(0)
    assert getattr(same, "_output_calibrator", None) is not None


def test_export_metadata_includes_calibration():
    ens, oof_matrix, y, _ = _build_ensemble_with_oof()
    md_off = ens.export_metadata()
    assert md_off["calibrate_output"] is False
    assert md_off["output_calibration"] is None
    ens.fit_output_calibrator(oof_matrix, y, method="linear")
    md_on = ens.export_metadata()
    assert md_on["calibrate_output"] is True
    assert md_on["output_calibration"]["method"] == "linear"


# --------------------------------------------------------------------------
# biz_value: S-shaped miscalibration -> isotonic wins on holdout
# --------------------------------------------------------------------------


def test_biz_val_isotonic_recalibration_lowers_holdout_rmse():
    """On a synthetic where the raw ensemble blend is S-shaped miscalibrated,
    fitting an isotonic calibrator on OOF and applying it on an INDEPENDENT
    holdout lowers RMSE by a clear margin AND straightens the calibration curve.

    Leakage-free by construction: the calibrator is fit on the OOF split only;
    the win is measured on a disjoint holdout the calibrator never saw.
    Measured isotonic holdout RMSE ~0.55x the raw blend; floor at 0.85x (15%).
    """
    rng = np.random.default_rng(2024)
    n = 6000
    y_all = rng.uniform(0, 10, size=n)
    blend_all = _s_shaped_miscal(y_all) + rng.normal(0, 0.15, size=n)
    c0_all = blend_all + rng.normal(0, 0.4, size=n)
    c1_all = 2.0 * blend_all - c0_all  # mean == blend exactly

    # OOF split (fit calibrator) vs holdout split (measure win) -- disjoint.
    oof_idx = np.arange(0, n // 2)
    hold_idx = np.arange(n // 2, n)

    ens = CompositeCrossTargetEnsemble.from_uniform_weights(
        [_ConstColModel(c0_all[hold_idx]), _ConstColModel(c1_all[hold_idx])],
        ["c0", "c1"],
    )
    X_hold = np.zeros((len(hold_idx), 1))
    raw_hold = ens.predict(X_hold)
    rmse_raw = _rmse(raw_hold, y_all[hold_idx])

    # Fit the calibrator strictly on the OOF half.
    oof_matrix = np.column_stack([c0_all[oof_idx], c1_all[oof_idx]])
    ens.fit_output_calibrator(oof_matrix, y_all[oof_idx], method="isotonic")
    cal_hold = ens.predict(X_hold)
    rmse_cal = _rmse(cal_hold, y_all[hold_idx])

    assert rmse_cal <= 0.85 * rmse_raw, (
        f"isotonic OOF recalibration should cut holdout RMSE to <=0.85x; raw={rmse_raw:.4f}, cal={rmse_cal:.4f} (ratio={rmse_cal / rmse_raw:.3f})"
    )

    # Calibration-curve straightness: slope of (binned mean pred vs binned mean
    # truth) should be closer to 1.0 after recalibration.
    def _curve_slope(pred):
        order = np.argsort(pred)
        bins = np.array_split(order, 10)
        px = np.array([pred[b].mean() for b in bins])
        ty = np.array([y_all[hold_idx][b].mean() for b in bins])
        a, _ = np.polyfit(px, ty, 1)
        return abs(a - 1.0)

    assert _curve_slope(cal_hold) < _curve_slope(raw_hold), "recalibration should straighten the calibration curve (slope closer to 1)"


def test_biz_val_linear_corrects_affine_shrinkage_bias():
    """NNLS-style shrinkage leaves an affine (scale<1) bias; a LINEAR calibrator
    recovers it. Holdout RMSE after linear recalibration <= 0.5x the shrunk
    blend's RMSE on a disjoint holdout (measured ~0.05x; floor very loose)."""
    rng = np.random.default_rng(99)
    n = 4000
    y_all = rng.uniform(-5, 15, size=n)
    # Shrunk + offset blend: pred = 0.6*y + 2 (classic regression-to-mean bias).
    blend_all = 0.6 * y_all + 2.0 + rng.normal(0, 0.2, size=n)
    c0_all = blend_all + rng.normal(0, 0.3, size=n)
    c1_all = 2.0 * blend_all - c0_all

    oof_idx = np.arange(0, n // 2)
    hold_idx = np.arange(n // 2, n)
    ens = CompositeCrossTargetEnsemble.from_uniform_weights(
        [_ConstColModel(c0_all[hold_idx]), _ConstColModel(c1_all[hold_idx])],
        ["c0", "c1"],
    )
    X_hold = np.zeros((len(hold_idx), 1))
    rmse_raw = _rmse(ens.predict(X_hold), y_all[hold_idx])
    oof_matrix = np.column_stack([c0_all[oof_idx], c1_all[oof_idx]])
    ens.fit_output_calibrator(oof_matrix, y_all[oof_idx], method="linear")
    rmse_cal = _rmse(ens.predict(X_hold), y_all[hold_idx])
    assert rmse_cal <= 0.5 * rmse_raw, f"linear recalibration should recover affine shrinkage bias; raw={rmse_raw:.4f}, cal={rmse_cal:.4f}"


# --------------------------------------------------------------------------
# biz_value / verdict-pin: isotonic is the most-accurate calibration DEFAULT
# even at small OOF (the regime where the 2-param sigmoid was hypothesised to
# generalise better). qual-15 shootout REJECTED that hypothesis: isotonic beats
# sigmoid on honest holdout RMSE in 8/8 seeds at every OOF size down to n=60.
# bench: composite/ensemble/_benchmarks/bench_calibration_method.py.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_oof", [60, 150, 400])
def test_biz_val_isotonic_beats_sigmoid_on_small_oof_holdout(n_oof):
    """The OutputCalibrator default ``method='isotonic'`` is the most-accurate
    map on an INDEPENDENT honest holdout even at the smallest OOF the cross-target
    ensemble uses, contradicting the "sigmoid is more robust at small n" prior.

    On an S-shaped (tanh-squashed) miscalibration the free-form isotonic PAV map
    tracks the distortion that the 2-parameter logistic-link sigmoid cannot, and
    the holdout-RMSE gap WIDENS as OOF shrinks (sigmoid's logit-OLS fit degrades
    faster than isotonic's). Measured isotonic/sigmoid holdout RMSE ratio ~0.49
    at n_oof=60; floor the win at <=0.85 (15% margin)."""
    rng = np.random.default_rng(7000 + n_oof)
    n_hold = 4000

    def _surface(m):
        s = rng.normal(0.0, 1.0, size=m)
        y = 2.0 * s + rng.normal(0.0, 0.3, size=m)
        p = 3.0 * np.tanh(0.8 * s) + rng.normal(0.0, 0.1, size=m)
        return p.astype(np.float64), y.astype(np.float64)

    p_oof, y_oof = _surface(n_oof)
    p_hold, y_hold = _surface(n_hold)

    iso = OutputCalibrator(method="isotonic").fit(p_oof, y_oof)
    sig = OutputCalibrator(method="sigmoid").fit(p_oof, y_oof)
    rmse_iso = _rmse(iso.predict(p_hold), y_hold)
    rmse_sig = _rmse(sig.predict(p_hold), y_hold)

    assert rmse_iso <= 0.85 * rmse_sig, (
        f"isotonic must beat sigmoid on the n_oof={n_oof} honest holdout (iso={rmse_iso:.4f}, sig={rmse_sig:.4f}, ratio={rmse_iso / rmse_sig:.3f})"
    )


# --------------------------------------------------------------------------
# biz_value / verdict-pin: the sigmoid OPTION now uses a proper maximum-likelihood
# Platt fit (qual-16), not the legacy OLS-on-centred-logit surrogate. On a genuine
# logistic miscalibration of a probability target -- the regime a sigmoid is the
# right tool for -- the MLE Platt map beats BOTH the raw blend AND the old OLS-logit
# map on an INDEPENDENT honest holdout. (The default method stays isotonic; this
# pins that the sigmoid OPTION is now actually useful.)
# bench: composite/ensemble/_benchmarks/bench_sigmoid_platt_vs_ols_logit.py.
# --------------------------------------------------------------------------


def _logistic_miscal_surface(m, rng):
    """Probability target + a logistic raw blend with the WRONG slope/offset."""
    s = rng.normal(0.0, 1.5, size=m)
    ptrue = 1.0 / (1.0 + np.exp(-s))
    y = (rng.uniform(size=m) < ptrue).astype(np.float64)
    raw = 1.0 / (1.0 + np.exp(-(0.4 * s - 0.5)))
    return raw.astype(np.float64), y


@pytest.mark.parametrize("n_oof", [60, 150, 400])
def test_biz_val_platt_sigmoid_beats_ols_logit_and_raw_on_holdout(n_oof):
    """The Platt (MLE) sigmoid_fit beats the legacy OLS-logit map AND the raw blend
    on a disjoint honest holdout for a genuinely-logistic probability miscalibration,
    at every OOF size down to n=60. Measured platt/ols_logit RMSE ratio ~0.92 and
    platt/raw ~0.92 at n_oof=60; floor the wins at <=0.98 (loose, regression-catching)."""
    rng = np.random.default_rng(9100 + n_oof)
    p_oof, y_oof = _logistic_miscal_surface(n_oof, rng)
    p_hold, y_hold = _logistic_miscal_surface(4000, rng)

    platt = OutputCalibrator(method="sigmoid", sigmoid_fit="platt").fit(p_oof, y_oof)
    ols = OutputCalibrator(method="sigmoid", sigmoid_fit="ols_logit").fit(p_oof, y_oof)
    rmse_platt = _rmse(platt.predict(p_hold), y_hold)
    rmse_ols = _rmse(ols.predict(p_hold), y_hold)
    rmse_raw = _rmse(p_hold, y_hold)

    assert rmse_platt <= 0.98 * rmse_ols, (
        f"MLE Platt must beat the legacy OLS-logit sigmoid at n_oof={n_oof} (platt={rmse_platt:.4f}, ols_logit={rmse_ols:.4f})"
    )
    assert rmse_platt <= 0.98 * rmse_raw, f"MLE Platt must beat the raw blend at n_oof={n_oof} (platt={rmse_platt:.4f}, raw={rmse_raw:.4f})"


def test_default_sigmoid_fit_is_platt():
    """The shipped sigmoid_fit default is the maximum-likelihood Platt map (qual-16).
    A future flip back to the OLS-logit surrogate must trip this sensor."""
    assert OutputCalibrator(method="sigmoid").sigmoid_fit == "platt"
    assert OutputCalibrator(method="sigmoid").export().get("method") == "sigmoid"


def test_sigmoid_platt_is_monotone_and_rejects_bad_fit():
    rng = np.random.default_rng(11)
    p_oof, y_oof = _logistic_miscal_surface(800, rng)
    cal = OutputCalibrator(method="sigmoid", sigmoid_fit="platt").fit(p_oof, y_oof)
    grid = np.linspace(p_oof.min(), p_oof.max(), 200)
    assert np.all(np.diff(cal.predict(grid)) >= -1e-9), "Platt sigmoid map not monotone"
    with pytest.raises(ValueError):
        OutputCalibrator(method="sigmoid", sigmoid_fit="bogus")


def test_ols_logit_sigmoid_fit_still_reachable_for_replay():
    """REJECTED != DELETED: the legacy OLS-logit fit stays opt-in for byte-identical replay."""
    rng = np.random.default_rng(13)
    p_oof, y_oof = _logistic_miscal_surface(500, rng)
    legacy = OutputCalibrator(method="sigmoid", sigmoid_fit="ols_logit").fit(p_oof, y_oof)
    assert legacy.sigmoid_fit == "ols_logit"
    assert legacy.export()["sigmoid"]["fit"] == "ols_logit"
    # Distinct parameters from the MLE fit (different objective).
    platt = OutputCalibrator(method="sigmoid", sigmoid_fit="platt").fit(p_oof, y_oof)
    assert not np.isclose(legacy._sig_A, platt._sig_A)


def test_default_calibration_method_is_isotonic():
    """The shipped default for both the OutputCalibrator and the cross-target
    discovery config stays ``isotonic`` (qual-15 verdict). A future flip to
    sigmoid/linear must trip this sensor + re-run the shootout bench."""
    assert OutputCalibrator().method == "isotonic"
    from mlframe.training._composite_target_discovery_config import (
        CompositeTargetDiscoveryConfig,
    )

    assert CompositeTargetDiscoveryConfig().cross_target_calibration_method == "isotonic"
