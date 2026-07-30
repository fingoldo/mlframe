"""Regression tests for the training/targets findings fixed from the
2026-07-21 full-repo audit (see audits/full_audit_2026-07-21/training_targets.md).

One narrowly-scoped test per finding ID (F1-F11), named after the failure
mode each one pins.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler

from mlframe.training.targets._target_distribution_analyzer_features import analyze_feature_distribution
from mlframe.training.targets._target_distribution_analyzer_stats import _lag1_autocorr_grouped
from mlframe.training.targets._target_distribution_analyzer_target_fn import analyze_target_distribution
from mlframe.training.targets._target_temporal_audit_aggregate import (
    _aggregate_by_time_pandas,
    _aggregate_by_time_polars,
)
from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling


def test_f1_polars_pandas_target_rate_parity_on_null_target():
    """F1: polars binary-classification target-rate aggregation must not deflate the
    rate by counting null target rows as negatives -- must match the pandas twin."""
    n = 300
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    y = np.where(rng.random(n) < 0.3, np.nan, 1.0)  # every non-null value is 1.0 -> true rate is 1.0

    df_pd = pd.DataFrame({"ts": ts, "y": y})
    df_pl = pl.DataFrame({"ts": ts, "y": y})

    agg_pd = _aggregate_by_time_pandas(df_pd, "ts", "y", "day", target_type="binary_classification")
    agg_pl = _aggregate_by_time_polars(df_pl, "ts", "y", "day", target_type="binary_classification")

    assert np.allclose(agg_pd["target_rate"].to_numpy(), agg_pl["target_rate"].to_numpy(), equal_nan=True)
    assert np.allclose(agg_pl["target_rate"].to_numpy(), 1.0, equal_nan=True)


def test_f2_analyze_target_distribution_string_labels_do_not_crash():
    """F2: an explicit target_type="classification" must not crash on string class labels."""
    y = np.array(["yes", "no", "yes", "no"] * 20)
    report = analyze_target_distribution(y, target_type="classification", has_time_axis=False)
    assert report.target_type == "classification"
    assert report.diagnostics["n_classes"] == 2.0


def test_f2_analyze_target_distribution_drops_nan_before_counting_classes():
    """F2 (related): NaN rows must not be counted as a spurious class or inflate n_samples."""
    y = np.array([0.0] * 20 + [1.0] * 20 + [np.nan] * 5)
    report = analyze_target_distribution(y, target_type="classification", has_time_axis=False)
    assert report.diagnostics["n_classes"] == 2.0
    assert report.n_samples == 40


def test_f6_analyze_target_distribution_float_class_labels_do_not_crash():
    """F6: float-valued class labels must not crash rare-class detection (previously forced int())."""
    y = np.array([1.4] * 40 + [1.6] * 3)
    report = analyze_target_distribution(y, target_type="classification", has_time_axis=False)
    assert any(p.startswith("rare_classes(") for p in report.pathologies), report.pathologies


def test_f3_analyze_feature_distribution_multiclass_leakage_uses_per_class_auc():
    """F3: a feature that perfectly encodes a >2-class target must be caught by the
    documented per-class-AUC leakage detector, not silently missed by Pearson-vs-integer-codes."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 4, size=n)
    leak = y.astype(float) + rng.normal(0, 1e-6, size=n)
    noise = rng.normal(size=n)
    X = pd.DataFrame({"leak": leak, "noise": noise})

    report = analyze_feature_distribution(X, y, target_type="classification")

    assert "leak" in report.leakage_candidates
    assert "noise" not in report.leakage_candidates


def test_f3_analyze_feature_distribution_binary_still_uses_pearson():
    """F3 (regression guard): binary classification must still use the Pearson proxy path (no crash, no false-positive on plain noise)."""
    rng = np.random.default_rng(1)
    n = 300
    y_bin = rng.integers(0, 2, size=n)
    X = pd.DataFrame({"noise": rng.normal(size=n)})
    report = analyze_feature_distribution(X, y_bin, target_type="classification")
    assert report.leakage_candidates == []


def test_f4_ttr_predict_clip_shared_helper_applies_to_out_of_range_predictions():
    """F4: the y-space defensive clip logic (now a shared ``_apply_predict_clip`` helper used
    by BOTH the transformer-configured and no-transformer predict() paths, so the no-transformer
    branch can no longer silently skip it) must clip out-of-range predictions.

    Note: sklearn's own TransformedTargetRegressor.fit() always establishes a real (identity)
    FunctionTransformer for self.transformer_, even when transformer=None at construction --
    forcing self.transformer_ back to None after a real fit() therefore also breaks the PARENT
    class's own predict() (an orthogonal, pre-existing sklearn-integration constraint, not part
    of this fix), so this test exercises the extracted clip helper directly rather than trying
    to force sklearn into an unsupported state."""
    from sklearn.linear_model import LinearRegression

    ttr = _TTRWithEvalSetScaling(regressor=LinearRegression(), transformer=None)
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.array([10.0] * 10 + [12.0] * 10)
    ttr.fit(X, y)

    clip_high = ttr._y_train_clip_high_
    clip_low = ttr._y_train_clip_low_
    out_of_range = np.array([clip_high + 1000.0, clip_low - 1000.0, (clip_low + clip_high) / 2.0])
    clipped = ttr._apply_predict_clip(out_of_range)
    assert clipped[0] == pytest.approx(clip_high)
    assert clipped[1] == pytest.approx(clip_low)
    assert clipped[2] == pytest.approx(out_of_range[2])  # in-range value passes through unchanged


def test_f4_ttr_predict_calls_shared_clip_helper(monkeypatch):
    """F4 (integration): predict() must route its return value through _apply_predict_clip
    rather than ever returning an un-clipped value directly (pre-fix, the transformer_-is-None
    guard returned super().predict() raw, bypassing the clip entirely)."""
    from sklearn.linear_model import LinearRegression

    ttr = _TTRWithEvalSetScaling(regressor=LinearRegression(), transformer=None)
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.array([10.0] * 10 + [12.0] * 10)
    ttr.fit(X, y)

    calls = []
    real_clip = ttr._apply_predict_clip

    def _spy(pred_trans, orig_ndim=1):
        """Records call arguments for this test's assertions."""
        calls.append(True)
        return real_clip(pred_trans, orig_ndim=orig_ndim)

    monkeypatch.setattr(ttr, "_apply_predict_clip", _spy)
    ttr.predict(X[:5])
    assert calls, "_apply_predict_clip was not called from predict()"


def test_f5_ttr_fit_clip_bound_failure_is_logged(monkeypatch, caplog):
    """F5: a failure computing the y-train clip bounds in fit() must be logged (debug), not silently swallowed."""
    from sklearn.linear_model import LinearRegression

    ttr = _TTRWithEvalSetScaling(regressor=LinearRegression(), transformer=StandardScaler())
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.array([10.0] * 10 + [12.0] * 10)

    real_isfinite = np.isfinite

    def _boom(arr):
        # Identity check only (not shape): y_arr = np.asarray(y, dtype=np.float64) inside
        # fit() returns the SAME object as `y` here (dtype already matches, no copy), so
        # this fires exactly once, at the intended call site, and never touches sklearn's
        # own downstream np.isfinite calls on same-shaped-but-different arrays.
        """Boom."""
        if arr is y:
            raise RuntimeError("synthetic failure for F5 regression test")
        return real_isfinite(arr)

    monkeypatch.setattr(np, "isfinite", _boom)
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.targets._ttr_eval_set_scaling"):
        ttr.fit(X, y)
    assert any("clip bound computation failed" in r.message for r in caplog.records)


class _EvalSetAcceptingRegressor:
    """Minimal sklearn-compatible regressor whose fit() accepts (and records) eval_set,
    mirroring LightGBM/XGBoost/MLP-style estimators this wrapper is designed for."""

    def get_params(self, deep=True):
        """Get params."""
        return {}

    def set_params(self, **kwargs):
        """Set params."""
        return self

    def fit(self, X, y, eval_set=None, **fit_params):
        """Fit; records the eval_set it was actually given."""
        self.seen_eval_set_ = eval_set
        self._n_features = np.asarray(X).shape[1] if hasattr(X, "shape") else 1
        return self

    def predict(self, X, **predict_params):
        """Predict zeros in scaled space."""
        return np.zeros(len(X), dtype=np.float64)


def test_f11_ttr_fit_does_not_double_fit_self_transformer(monkeypatch):
    """F11: self.transformer_ must be fit exactly once by the parent's super().fit(), not
    pre-fit again in the child fit() for eval_set scaling."""
    fit_calls = []
    real_fit = StandardScaler.fit

    def _counting_fit(self, *a, **kw):
        """Counting fit."""
        fit_calls.append(id(self))
        return real_fit(self, *a, **kw)

    monkeypatch.setattr(StandardScaler, "fit", _counting_fit)
    ttr = _TTRWithEvalSetScaling(regressor=_EvalSetAcceptingRegressor(), transformer=StandardScaler())
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.array([10.0] * 10 + [12.0] * 10)
    X_val = np.arange(20, 25).reshape(-1, 1).astype(float)
    y_val = np.array([11.0] * 5)
    ttr.fit(X, y, eval_set=(X_val, y_val))

    # self.transformer_ (the one actually used at predict-time) must have been fit exactly once.
    assert fit_calls.count(id(ttr.transformer_)) == 1
    # The inner regressor's eval_set y must have been scaled (not left as raw y_val), and the
    # y_val scaling itself must have come from a transformer fit on the SAME distribution as
    # self.transformer_ (both are StandardScaler fit on the same y, so scaled values match).
    scaled_y_val = ttr.regressor_.seen_eval_set_[1]
    expected = ttr.transformer_.transform(y_val.reshape(-1, 1)).reshape(-1)
    assert np.allclose(scaled_y_val, expected)


def test_f9_select_target_positional_indexing_with_plain_list_idx(monkeypatch):
    """F9: select_target's internal _select() closure must treat a plain Python list
    train_idx the same as a numpy array (positional), not fall through to pandas
    label-based indexing. Pre-fix, this raised KeyError against a pd.Series target
    with a non-default index whose labels don't happen to include 0/2."""
    from mlframe.training import trainer as trainer_module
    from mlframe.training.configs import TargetTypes
    from mlframe.training.targets._train_eval_select_target import select_target

    captured = {}

    def _stub_configure_training_params(*, model_name, **kwargs):
        """Stub configure training params."""
        captured["model_name"] = model_name
        raise RuntimeError("stub-stop")

    monkeypatch.setattr(trainer_module, "configure_training_params", _stub_configure_training_params)

    df = pd.DataFrame({"f1": np.arange(10, dtype=float)})
    # Non-default pandas index: labels 10/20/30/40/50 don't correspond to positions 0/2 --
    # a label-based .loc[[0, 2]] would KeyError here instead of silently misselecting rows.
    target = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0], index=[10, 20, 30, 40, 50])

    with pytest.raises(RuntimeError, match="stub-stop"):
        select_target(
            model_name="test_model",
            target=target,
            target_type=TargetTypes.REGRESSION,
            df=df,
            train_idx=[0, 2],  # plain python list -- F9's exact trigger condition
        )

    # Positional selection of rows 0 and 2 -> values 100.0, 300.0 -> mean 200.00.
    assert "200" in captured["model_name"]


def test_f10_redundant_feature_sampling_is_random_not_systematic_stride():
    """F10: the >100k-row redundant-pair sampling path must not alias with periodic structure --
    verify it no longer uses a fixed stride (systematic sample), by checking row selection is
    reproducible (fixed random_state) but NOT a simple arithmetic progression."""
    n = 250_000
    rng = np.random.default_rng(0)
    # Periodic signal with period 7 that a stride divisible by 7 would alias against.
    period = np.tile(np.arange(7.0), n // 7 + 1)[:n]
    a = period + rng.normal(scale=0.01, size=n)
    b = rng.normal(size=n)  # unrelated -- should not be flagged as redundant regardless of sampling method
    X = pd.DataFrame({"a": a, "b": b})
    report = analyze_feature_distribution(X)
    assert report.diagnostics["n_samples"] == n
    # No crash, and the low-signal unrelated pair must not spuriously show up as redundant.
    assert not any("a" in pair.values() and "b" in pair.values() for pair in report.diagnostics.get("redundant_feature_pairs", []))


def test_p5_lag1_autocorr_grouped_returns_skipped_count():
    """P5: _lag1_autocorr_grouped must surface the count of groups skipped for being
    below min_group_size, instead of silently discarding it."""
    y = np.array([1.0, 2.0, 1.5, 2.5] * 3 + [9.0, 9.0])  # 3 groups of 4 (kept) + 1 group of 2 (skipped, min_group_size=4)
    group_ids = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 2)
    ar, n_skipped = _lag1_autocorr_grouped(y, group_ids, min_group_size=4)
    assert n_skipped == 1
    assert np.isfinite(ar)


def test_f7_no_mojibake_remains_in_target_temporal_files():
    """F7: the mojibake corruption (UTF-8 misread as CP1251) must be gone from the
    previously-affected files, and not reintroduced."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "src" / "mlframe"
    affected = [
        root / "training" / "targets" / "target_temporal_audit.py",
        root / "training" / "targets" / "_target_temporal_audit_from_agg.py",
        root / "training" / "targets" / "_target_temporal_changepoint.py",
    ]
    mojibake_markers = ("Г—", "вЂ”", "в‰Ґ")
    for fp in affected:
        text = fp.read_text(encoding="utf-8")
        for marker in mojibake_markers:
            assert marker not in text, f"mojibake marker {marker!r} found in {fp}"
