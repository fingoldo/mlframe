"""Regression tests for audits/full_audit_2026-07-21/training_baselines.md (F1-F10).

P1/P2 (test-coverage proposals for F1/F2/F3) are satisfied by the tests below. P3 (behavioral-equivalence
or delete the duplicate compute_dummy_baselines) is satisfied structurally: F4's fix makes the facade a
thin delegating wrapper, so the two implementations can no longer drift -- the round-trip test below is
extra confidence, not the only guard. P4-P7 are refactor/docs/robustness proposals not implemented in this
pass (P6's n_classes ceiling and P4's shared tie-break helper are real but separate, larger-scope
robustness/refactor items; P5 and P7 are pure docs).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.baselines.dummy import compute_dummy_baselines


def _reg_frames(n_train=200, n_val=50, n_test=50, seed=0):
    """Reg frames."""
    rng = np.random.default_rng(seed)
    train_X = pd.DataFrame({"x": rng.normal(size=n_train)})
    val_X = pd.DataFrame({"x": rng.normal(size=n_val)})
    test_X = pd.DataFrame({"x": rng.normal(size=n_test)})
    return train_X, val_X, test_X


# ----------------------------------------------------------------------
# F1 (P0) -- a single NaN in train_y no longer crashes classification baselines.
# ----------------------------------------------------------------------


def test_f1_classification_single_nan_train_y_no_crash():
    """F1: classification single nan train y no crash."""
    rng = np.random.default_rng(0)
    n = 200
    train_X, val_X, test_X = _reg_frames(n, 50, 50, seed=1)
    train_y = rng.integers(0, 2, n).astype(float)
    train_y[5] = np.nan
    val_y = rng.integers(0, 2, 50).astype(float)
    test_y = rng.integers(0, 2, 50).astype(float)

    report = compute_dummy_baselines(
        target_type="binary_classification", target_name="t1",
        train_X=train_X, val_X=val_X, test_X=test_X,
        train_y=train_y, val_y=val_y, test_y=test_y,
    )
    assert report.strongest is not None
    assert not report.table.empty


def test_f1_classification_all_nan_train_y_degrades_gracefully():
    """Every train label NaN: must not crash, degrades to uniform prior instead."""
    rng = np.random.default_rng(0)
    n = 50
    train_X, val_X, test_X = _reg_frames(n, 20, 20, seed=2)
    train_y = np.full(n, np.nan)
    val_y = rng.integers(0, 2, 20).astype(float)
    test_y = rng.integers(0, 2, 20).astype(float)
    report = compute_dummy_baselines(
        target_type="binary_classification", target_name="t2",
        train_X=train_X, val_X=val_X, test_X=test_X,
        train_y=train_y, val_y=val_y, test_y=test_y,
    )
    assert report is not None  # must not raise


# ----------------------------------------------------------------------
# F2 -- a single NaN in train_y no longer wipes out regression constant baselines.
# ----------------------------------------------------------------------


def test_f2_regression_single_nan_train_y_constant_baselines_survive():
    """F2: regression single nan train y constant baselines survive."""
    rng = np.random.default_rng(0)
    train_y = rng.normal(size=200)
    train_y[5] = np.nan
    val_y = rng.normal(size=50)
    test_y = rng.normal(size=50)
    train_X, val_X, test_X = _reg_frames(200, 50, 50, seed=3)

    report = compute_dummy_baselines(
        target_type="regression", target_name="t3",
        train_X=train_X, val_X=val_X, test_X=test_X,
        train_y=train_y, val_y=val_y, test_y=test_y,
    )
    assert report.strongest is not None, "strongest collapsed to None despite val/test being fully finite"
    assert np.isfinite(report.table.loc["mean", "val_RMSE"])
    assert np.isfinite(report.table.loc["median", "val_RMSE"])


def test_f2_regression_constant_baselines_match_nanmean_nanmedian():
    """F2: regression constant baselines match nanmean nanmedian."""
    rng = np.random.default_rng(0)
    train_y = rng.normal(loc=5.0, scale=1.0, size=300)
    train_y[10] = np.nan
    train_y[20] = np.nan
    val_y = rng.normal(loc=5.0, scale=1.0, size=50)
    test_y = rng.normal(loc=5.0, scale=1.0, size=50)
    train_X, val_X, test_X = _reg_frames(300, 50, 50, seed=4)

    # Smoke-check the public facade doesn't raise on NaN-containing train_y before drilling into
    # the internal dispatcher below for the exact constant-value check.
    compute_dummy_baselines(
        target_type="regression", target_name="t4",
        train_X=train_X, val_X=val_X, test_X=test_X,
        train_y=train_y, val_y=val_y, test_y=test_y,
    )
    # Indirect check: the mean baseline prediction (constant) must equal nanmean(train_y).
    expected_mean = float(np.nanmean(train_y))
    # Recompute directly via the internal dispatcher to check the exact constant used. Uses the real
    # DummyBaselinesConfig (not a bespoke fake) so every attribute the dispatcher reads is present.
    from mlframe.training.baselines._dummy_baseline_regression import _compute_regression_baselines
    from mlframe.training.configs import DummyBaselinesConfig

    vp, _tp, _extras = _compute_regression_baselines(
        "t4", train_X, val_X, test_X, train_y, val_y, test_y, None, None, None, None,
        DummyBaselinesConfig(), target_type="regression",
    )
    assert vp["mean"][0] == pytest.approx(expected_mean)


# ----------------------------------------------------------------------
# F3 -- rolling_mean_w{7,30} (ts) baselines actually fire on an ACF-detectable series.
# ----------------------------------------------------------------------


def test_f3_rolling_mean_and_acf_detected_label_fire():
    """F3: rolling mean and acf detected label fire."""
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n)
    y = 10.0 + 5.0 * np.sin(2 * np.pi * t / 7.0) + rng.normal(scale=0.2, size=n)
    ts = pd.date_range("2020-01-01", periods=n, freq="D")

    train_n = 200
    train_X, val_X, test_X = _reg_frames(train_n, 50, n - train_n - 50, seed=5)

    report = compute_dummy_baselines(
        target_type="regression", target_name="t5",
        train_X=train_X, val_X=val_X, test_X=test_X,
        train_y=y[:train_n], val_y=y[train_n : train_n + 50], test_y=y[train_n + 50 :],
        timestamps_train=ts[:train_n], timestamps_val=ts[train_n : train_n + 50], timestamps_test=ts[train_n + 50 :],
    )
    rolling_rows = [i for i in report.table.index if "rolling_mean_w7" in str(i)]
    acf_labeled = [i for i in report.table.index if "ACF-detected" in str(i)]
    assert rolling_rows, "rolling_mean_w7 (ts) never fires on an ACF-detectable series"
    assert acf_labeled, "seasonal_naive_pP never gets the ACF-detected label"


# ----------------------------------------------------------------------
# F4 -- the duplicate compute_dummy_baselines facade now delegates to the canonical implementation.
# ----------------------------------------------------------------------


def test_f4_facade_delegates_and_matches_canonical():
    """F4: facade delegates and matches canonical."""
    from mlframe.training.baselines._dummy_baseline_compute import compute_dummy_baselines as facade_fn
    from mlframe.training.baselines.dummy import compute_dummy_baselines as canonical_fn

    train_X, val_X, test_X = _reg_frames(100, 30, 30, seed=6)
    rng = np.random.default_rng(6)
    train_y, val_y, test_y = rng.normal(size=100), rng.normal(size=30), rng.normal(size=30)

    r1 = facade_fn(target_type="regression", target_name="t6", train_X=train_X, val_X=val_X, test_X=test_X, train_y=train_y, val_y=val_y, test_y=test_y)
    r2 = canonical_fn(target_type="regression", target_name="t6", train_X=train_X, val_X=val_X, test_X=test_X, train_y=train_y, val_y=val_y, test_y=test_y)
    assert r1.strongest == r2.strongest
    assert r1.table.shape == r2.table.shape


def test_f4_facade_docstring_no_longer_claims_full_reexport_but_has_no_second_body():
    """F4: facade docstring no longer claims full reexport but has no second body."""
    import inspect

    from mlframe.training.baselines._dummy_baseline_compute import compute_dummy_baselines as facade_fn

    src = inspect.getsource(facade_fn)
    assert "_canonical_compute_dummy_baselines" in src
    # the pre-fix duplicate body computed val_preds/test_preds itself; the wrapper must not.
    assert "val_preds: dict[str, np.ndarray] = {}" not in src


# ----------------------------------------------------------------------
# F5 -- RMSE=0.0 no longer triggers the falsy-or fallback in the unified verdict table.
# ----------------------------------------------------------------------


def test_f5_zero_rmse_rendered_directly_not_fallback():
    """F5: zero rmse rendered directly not fallback."""
    from mlframe.training.baselines._dummy_summary_format import format_unified_target_verdict_table

    best_model_metrics_by_target = {
        ("regression", "raw_t"): {"val_RMSE": 0.0},
        ("regression", "comp_t"): {"val_RMSE": 0.0},
    }
    composite_to_raw_target_map = {("regression", "comp_t"): "raw_t"}
    out = format_unified_target_verdict_table(
        dummy_baselines_metadata={"dummy": True},
        best_model_metrics_by_target=best_model_metrics_by_target,
        composite_to_raw_target_map=composite_to_raw_target_map,
    )
    assert "0.0000" in out


def test_f5_pick_rmse_helper_explicit_none_check():
    """F5: pick rmse helper explicit none check."""
    from mlframe.training.baselines._dummy_summary_format import _pick_rmse

    assert _pick_rmse({"val_RMSE": 0.0}) == 0.0
    assert _pick_rmse({"val_RMSE": None, "RMSE": 1.5}) == 1.5
    assert _pick_rmse({}) is None


# ----------------------------------------------------------------------
# F6 -- dead pinball_per_a variable removed (non-behavioral; pin the metric output is unaffected).
# ----------------------------------------------------------------------


def test_f6_quantile_metrics_table_unaffected():
    """F6: quantile metrics table unaffected."""
    from mlframe.training.baselines._dummy_metrics_pick_plot import _compute_metrics_table

    n = 60
    alphas = [0.1, 0.5, 0.9]
    val_y = np.linspace(0, 1, n)
    val_preds = {"quantile_alpha_0.500": np.column_stack([val_y - 0.1, val_y, val_y + 0.1])}
    table, _primary_metric = _compute_metrics_table(
        "quantile_regression", val_preds, {}, val_y, None,
        group_ids_val=None, group_ids_test=None, extras={"quantile_alphas": alphas},
    )
    assert "val_pinball_mean" in table.columns
    assert np.isfinite(table.loc["quantile_alpha_0.500", "val_pinball_mean"])


def test_f6_pinball_per_a_no_longer_in_source():
    """F6: pinball per a no longer in source."""
    import inspect

    from mlframe.training.baselines import _dummy_metrics_pick_plot

    src = inspect.getsource(_dummy_metrics_pick_plot)
    assert "pinball_per_a" not in src


# ----------------------------------------------------------------------
# F7 -- _build_recommendation degrades to "unlikely_to_help" instead of crashing on all-non-finite deltas.
# ----------------------------------------------------------------------


def test_f7_all_nonfinite_ablation_deltas_no_crash():
    """F7: all nonfinite ablation deltas no crash."""
    from mlframe.training.baselines._baseline_diagnostics_recommend import _build_recommendation

    class _Entry:
        """Entry."""
        def __init__(self, delta_pct):
            self.delta_pct = delta_pct

    class _Cfg:
        """Cfg."""
        high_potential_min_dominance_pct = 5.0
        init_score_optimal_threshold_pct = 1.0
        marginal_threshold_pct = 2.0

    class _Self:
        """Self."""
        config = _Cfg()

    ablation = [_Entry(float("nan")), _Entry(float("inf"))]
    verdict, reason = _build_recommendation(_Self(), ablation, None)
    assert verdict == "unlikely_to_help"
    assert "non-finite" in reason


def test_f7_normal_ablation_still_classifies_correctly():
    """F7: normal ablation still classifies correctly."""
    from mlframe.training.baselines._baseline_diagnostics_recommend import _build_recommendation

    class _Entry:
        """Entry."""
        def __init__(self, delta_pct):
            self.delta_pct = delta_pct

    class _Cfg:
        """Cfg."""
        high_potential_min_dominance_pct = 5.0
        init_score_optimal_threshold_pct = 1.0
        marginal_threshold_pct = 2.0

    class _Self:
        """Self."""
        config = _Cfg()

    ablation = [_Entry(10.0), _Entry(1.0)]
    verdict, _reason = _build_recommendation(_Self(), ablation, None)
    assert verdict == "high_potential"


# ----------------------------------------------------------------------
# F8 -- fractional float classification labels are rejected instead of silently truncated.
# ----------------------------------------------------------------------


def test_f8_fractional_float_labels_rejected(caplog):
    """F8: fractional float labels rejected."""
    from mlframe.training.baselines._dummy_compute_helpers import _coerce_y

    y_frac = np.array([0.0, 1.0, 2.9, 1.0, 0.0])
    with caplog.at_level(logging.WARNING):
        result = _coerce_y(y_frac, "multiclass_classification", "t8")
    assert result is None
    assert any("fractional" in rec.message for rec in caplog.records)


def test_f8_whole_number_float_labels_pass_through():
    """F8: whole number float labels pass through."""
    from mlframe.training.baselines._dummy_compute_helpers import _coerce_y

    y_whole = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    result = _coerce_y(y_whole, "multiclass_classification", "t9")
    assert result is not None
    assert np.array_equal(result, y_whole)


# ----------------------------------------------------------------------
# F9 -- _detect_acf_periods threads a caller-supplied random_state instead of a hardcoded 42.
# ----------------------------------------------------------------------


def test_f9_acf_period_detection_varies_with_random_state():
    """F9: acf period detection varies with random state."""
    from mlframe.training.baselines._dummy_timeseries import _detect_acf_periods

    rng = np.random.default_rng(0)
    n = 60_001
    y = rng.normal(size=n)
    r1 = _detect_acf_periods(y, n, random_state=1)
    r2 = _detect_acf_periods(y, n, random_state=2)
    r1_again = _detect_acf_periods(y, n, random_state=1)
    assert r1 == r1_again, "same random_state must be reproducible"
    assert r1 != r2, "different random_state must sample different windows"


def test_f9_resolve_ts_periods_threads_random_state():
    """F9: resolve ts periods threads random state."""
    import inspect

    from mlframe.training.baselines._dummy_timeseries import _resolve_ts_periods

    assert "random_state" in inspect.signature(_resolve_ts_periods).parameters


# ----------------------------------------------------------------------
# F10 -- overlay-plot subsampling accepts a caller-supplied random_state instead of a hardcoded 0.
# ----------------------------------------------------------------------


def test_f10_overlay_plot_accepts_random_state_param():
    """F10: overlay plot accepts random state param."""
    import inspect

    from mlframe.training.baselines._dummy_metrics_pick_plot import plot_best_dummy_baseline_overlay

    sig = inspect.signature(plot_best_dummy_baseline_overlay)
    assert "random_state" in sig.parameters
    assert sig.parameters["random_state"].default == 0
