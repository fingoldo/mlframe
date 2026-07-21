"""Regression tests for audits/full_audit_2026-07-21/training_composite_loose_a.md findings F1-F18.

F3 (docstring accuracy near F4's fix) has no separate test -- covered by ordinary import/syntax checks.
F5 (decision_function error message) and F13/F14 (docstring-only fixes) are pinned via direct string
checks below where cheap. F4 (stacking_multi_stage.py's X.copy() -> X.copy(deep=False)) was already
fixed by an earlier session-wide large-frame-copy sweep; this file adds its formal regression test.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# F1 (P1): drift monitor silently read "no drift" on an empty/all-non-finite batch
# ---------------------------------------------------------------------------


def test_f1_drift_monitor_alerts_on_all_nan_base_column():
    """A broken upstream feed (all-NaN base column) now alerts via base_missing[col], not silently PSI=0."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from mlframe.training.composite.estimator import CompositeTargetEstimator
    from mlframe.training.composite.monitoring import CompositeDriftMonitor

    rng = np.random.default_rng(0)
    n = 2000
    base = rng.normal(0.0, 1.0, n)
    feat = rng.normal(0.0, 1.0, n)
    y = base + 0.5 * feat + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"lag": base, "feat": feat})
    est = CompositeTargetEstimator(base_estimator=HistGradientBoostingRegressor(max_iter=20), transform_name="diff", base_column="lag")
    est.fit(X, y)
    monitor = CompositeDriftMonitor(est)

    n_batch = 500
    X_broken = pd.DataFrame({"lag": np.full(n_batch, np.nan), "feat": rng.normal(size=n_batch)})
    report = monitor.monitor(X_broken, reference=X, y_reference=y)
    assert report["signals"]["base_missing[lag]"]["alert"] is True
    assert report["alert"] is True

    X_ok = pd.DataFrame({"lag": rng.normal(size=n_batch), "feat": rng.normal(size=n_batch)})
    report_ok = monitor.monitor(X_ok)
    assert report_ok["signals"]["base_missing[lag]"]["alert"] is False


def test_f1_missing_data_fraction_helper():
    """_missing_data_fraction returns 1.0 for empty/all-non-finite and 0.0 for fully-finite arrays."""
    from mlframe.training.composite.monitoring import _missing_data_fraction

    assert _missing_data_fraction(np.array([])) == 1.0
    assert _missing_data_fraction(np.full(10, np.nan)) == 1.0
    assert _missing_data_fraction(np.array([1.0, 2.0, np.nan, 3.0])) == pytest.approx(0.25)
    assert _missing_data_fraction(np.array([1.0, 2.0, 3.0])) == 0.0


# ---------------------------------------------------------------------------
# F2 (P1): _booster_margin.py swallowed a REAL predict() failure on a matched booster
# ---------------------------------------------------------------------------


def test_f2_matched_booster_predict_failure_propagates():
    """A genuine predict() error on an isinstance-matched booster now propagates, instead of a misleading NotImplementedError."""
    import lightgbm as lgb

    from mlframe.training.composite._booster_margin import inner_raw_margin

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = rng.integers(0, 2, size=200)
    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    model.fit(X, y)

    X_bad = rng.normal(size=(50, 99))
    with pytest.raises(ValueError, match="features"):
        inner_raw_margin(model, X_bad, lgbm_attr="LGBMClassifier", xgb_attr="XGBClassifier", catboost_attr="CatBoostClassifier", wrapper_name="TestWrapper", keep_2d=False)


def test_f2_genuinely_unmatched_model_still_raises_not_implemented():
    """A model type that matches NO family still raises the documented NotImplementedError."""
    from mlframe.training.composite._booster_margin import inner_raw_margin

    class NotABooster:
        """Stub object matching none of the supported booster family types."""
        pass

    with pytest.raises(NotImplementedError, match="no raw-margin path"):
        inner_raw_margin(NotABooster(), None, lgbm_attr="LGBMClassifier", xgb_attr="XGBClassifier", catboost_attr="CatBoostClassifier", wrapper_name="TestWrapper", keep_2d=False)


# ---------------------------------------------------------------------------
# F3 (P1): batch_size was a stored-but-never-read constructor param (always full-batch)
# ---------------------------------------------------------------------------


def test_f3_multitask_auxiliary_loss_batch_size_changes_training():
    """MultiTaskAuxiliaryLossRegressor's batch_size now actually drives mini-batch SGD."""
    from mlframe.training.composite.multitask_auxiliary_loss import MultiTaskAuxiliaryLossRegressor

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n).astype(np.float32)

    m_full = MultiTaskAuxiliaryLossRegressor(n_epochs=10, batch_size=None, random_state=0)
    m_full.fit(X, y)
    m_mini = MultiTaskAuxiliaryLossRegressor(n_epochs=10, batch_size=32, random_state=0)
    m_mini.fit(X, y)

    assert len(m_full.train_losses_) == 10
    assert len(m_mini.train_losses_) == 10
    assert m_full.train_losses_ != m_mini.train_losses_


def test_f3_additive_decomposition_batch_size_changes_training():
    """AdditiveDecompositionRegressor's batch_size now actually drives mini-batch SGD."""
    from mlframe.training.composite.additive_decomposition import AdditiveDecompositionRegressor

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n).astype(np.float32)

    a_full = AdditiveDecompositionRegressor(component_names=("c1", "c2"), n_epochs=10, batch_size=None, random_state=0)
    a_full.fit(X, y)
    a_mini = AdditiveDecompositionRegressor(component_names=("c1", "c2"), n_epochs=10, batch_size=32, random_state=0)
    a_mini.fit(X, y)
    assert a_full.train_losses_ != a_mini.train_losses_


# ---------------------------------------------------------------------------
# F4 (P1, memory): stacking_multi_stage.py's _concat_meta pandas branch -- already fixed by an
# earlier repo-wide large-frame-copy sweep; formal regression test added here.
# ---------------------------------------------------------------------------


def test_f4_concat_meta_uses_shallow_copy_not_full_copy():
    """_concat_meta's pandas branch calls X.copy(deep=False), sharing buffers with the caller's frame."""
    import inspect

    from mlframe.training.composite.stacking_multi_stage import MultiStageMetaFeatureStacker

    src = inspect.getsource(MultiStageMetaFeatureStacker._concat_meta)
    assert "X.copy(deep=False)" in src
    assert "X.copy()" not in src

    X = pd.DataFrame({"a": np.arange(100, dtype=np.float64)})
    meta_cols = {"meta_1": np.arange(100, dtype=np.float64)}
    out = MultiStageMetaFeatureStacker._concat_meta(X, meta_cols)
    # Shallow copy: the untouched 'a' column's underlying buffer is shared with X's.
    assert np.shares_memory(out["a"].to_numpy(), X["a"].to_numpy())
    assert list(out["meta_1"]) == list(meta_cols["meta_1"])
    assert list(X.columns) == ["a"], "X itself must remain unmutated by _concat_meta"


# ---------------------------------------------------------------------------
# F5 (P2): decision_function's NotFittedError named the wrong method
# ---------------------------------------------------------------------------


def test_f5_decision_function_not_fitted_error_names_itself():
    """F5: decision function not fitted error names itself."""
    from mlframe.training.composite.classification import CompositeClassificationEstimator
    from sklearn.exceptions import NotFittedError

    est = CompositeClassificationEstimator(base_estimator=None)
    with pytest.raises(NotFittedError, match="decision_function"):
        est.decision_function(pd.DataFrame({"a": [1.0]}))


# ---------------------------------------------------------------------------
# F6 (P2): calibration_report and _bin_top_label_calibration were two independent binning impls
# ---------------------------------------------------------------------------


def test_f6_calibration_binning_shared_between_classification_and_diagnostics():
    """Both callers now delegate to the SAME _calibration_binning.top_label_calibration_bins core."""
    from mlframe.training.composite._calibration_binning import top_label_calibration_bins
    from mlframe.training.composite.diagnostics import _bin_top_label_calibration

    rng = np.random.default_rng(0)
    n, k = 500, 3
    classes = np.array([0, 1, 2])
    proba = rng.dirichlet(np.ones(k), size=n)
    y_true = classes[np.argmax(proba + rng.normal(scale=0.3, size=(n, k)), axis=1)]

    direct = top_label_calibration_bins(y_true, proba, classes, n_bins=10)
    via_diag = _bin_top_label_calibration(y_true, proba, n_bins=10, sample_n=n, random_state=0)
    assert direct["ece"] == via_diag["ece"]
    assert np.array_equal(direct["bin_count"], via_diag["bin_count"])


def test_f6_calibration_report_end_to_end():
    """F6: calibration report end to end."""
    import lightgbm as lgb

    from mlframe.training.composite.classification import CompositeClassificationEstimator

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = (X["f1"] + 0.5 * X["f2"] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    est = CompositeClassificationEstimator(base_estimator=lgb.LGBMClassifier(n_estimators=20, verbose=-1))
    est.fit(X, y)
    report = est.calibration_report(X, y, n_bins=10)
    assert 0.0 <= report["ece"] <= 1.0
    assert report["bin_count"].sum() == n


# ---------------------------------------------------------------------------
# F7 (P2, perf): per_group_router.py ran the global fallback on ALL rows, wasting inference work
# ---------------------------------------------------------------------------


class _FakeSpec:
    """Fake Spec."""
    def __init__(self, transform_name, base_column):
        self.transform_name = transform_name
        self.base_column = base_column


class _FakeDiscovery:
    """Fake Discovery."""
    specs_ = [_FakeSpec("diff", "base")]
    specs_by_group_ = {"A": [_FakeSpec("diff", "base")], "B": [_FakeSpec("diff", "base")]}


def _fit_router():
    """Fit router."""
    from sklearn.linear_model import LinearRegression

    from mlframe.training.composite.per_group_router import PerGroupCompositeRouter

    rng = np.random.default_rng(0)
    n_per_group = 300
    groups = np.repeat(["A", "B", "C"], n_per_group)
    base = rng.normal(size=len(groups))
    feat = rng.normal(size=len(groups))
    y = base + 0.5 * feat + rng.normal(scale=0.1, size=len(groups))
    X = pd.DataFrame({"base": base, "feat": feat, "grp": groups})
    router = PerGroupCompositeRouter(discovery=_FakeDiscovery(), base_estimator=LinearRegression(), group_column="grp", min_group_fit_rows=10)
    router.fit(X, y)
    return router, X, groups, n_per_group


def test_f7_global_fallback_only_runs_on_unrouted_rows():
    """F7: global fallback only runs on unrouted rows."""
    router, X, _groups, n_per_group = _fit_router()
    orig_predict = router.global_estimator_.predict
    calls = []

    def spy_predict(X_arg):
        """Records predict() calls for this test's assertions."""
        calls.append(len(X_arg))
        return orig_predict(X_arg)

    router.global_estimator_.predict = spy_predict
    preds = router.predict(X)
    assert calls == [n_per_group]
    assert not np.isnan(preds).any()


def test_f7_predictions_unchanged_by_the_perf_fix():
    """F7: predictions unchanged by the perf fix."""
    router, X, groups, _n_per_group = _fit_router()
    preds = router.predict(X)
    mask_a = groups == "A"
    direct_a = router.group_estimators_["A"].predict(X.loc[mask_a].drop(columns=["grp"]))
    assert np.allclose(preds[mask_a], direct_a)


# ---------------------------------------------------------------------------
# F8 (P2, docs): survival predict() docstring overclaimed "median" for the aware-censoring mode
# ---------------------------------------------------------------------------


def test_f8_survival_predict_docstring_no_longer_overclaims_median():
    """F8: survival predict docstring no longer overclaims median."""
    from mlframe.training.composite.survival import CompositeSurvivalEstimator

    doc = CompositeSurvivalEstimator.predict.__doc__ or ""
    assert "observed_only" in doc
    assert "not a calibrated median" in doc or "not a true median" in doc


# ---------------------------------------------------------------------------
# F9 (P2): autoconfig rationale described an action ("added transforms") that didn't happen
# ---------------------------------------------------------------------------


def test_f9_autoconfig_rationale_omits_transforms_when_none_added():
    """F9: autoconfig rationale omits transforms when none added."""
    import mlframe.training.configs as configs_mod
    from mlframe.training.composite.autoconfig import suggest_discovery_config

    orig_cls = configs_mod.CompositeTargetDiscoveryConfig

    class PatchedConfig(orig_cls):
        """Subclass that force-appends 'signed_power_y' to the discovered transforms list."""
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if "signed_power_y" not in self.transforms:
                object.__setattr__(self, "transforms", [*list(self.transforms), "signed_power_y"])

    configs_mod.CompositeTargetDiscoveryConfig = PatchedConfig
    try:
        rng = np.random.default_rng(0)
        n = 2000
        y = rng.lognormal(mean=0.0, sigma=1.5, size=n)
        df = pd.DataFrame({"y": y, "f1": rng.normal(size=n)})
        _cfg, rationale = suggest_discovery_config(df, "y", ["f1"])
        assert "transforms" not in rationale
    finally:
        configs_mod.CompositeTargetDiscoveryConfig = orig_cls


# ---------------------------------------------------------------------------
# F10 (P2): serving.py silently substituted 0.0 for a non-finite y_train_median, no warning
# ---------------------------------------------------------------------------


def test_f10_load_serving_spec_warns_on_nonfinite_y_train_median(caplog):
    """F10: load serving spec warns on nonfinite y train median."""
    from mlframe.training.composite.serving import load_serving_spec

    spec = {
        "transform_name": "additive_residual",
        "fitted_params": {
            "y_train_median": float("nan"), "t_clip_low": float("-inf"), "t_clip_high": float("inf"),
            "y_clip_low": float("-inf"), "y_clip_high": float("inf"),
        },
        "fallback_predict": "y_train_median",
        "multi_base": False,
    }
    with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.serving"):
        load_serving_spec(spec)
    assert any("non-finite" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# F11 (P2): chained_window_forecast.py's growth_ratio had no zero-guard on chain_mse[0]
# ---------------------------------------------------------------------------


def test_f11_error_accumulation_handles_zero_baseline_mse():
    """F11: error accumulation handles zero baseline mse."""
    from sklearn.linear_model import LinearRegression

    from mlframe.training.composite.chained_window_forecast import ChainedWindowForecaster

    forecaster = ChainedWindowForecaster(stage1_estimator=LinearRegression(), stage2_estimator=LinearRegression())
    targets = [np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.5, 2.5])]
    preds = [np.array([1.0, 2.0]), np.array([1.1, 2.1]), np.array([1.4, 2.6])]
    call_idx = [0]

    def fake_predict(X):
        """Fake predict callable returning a fixed array."""
        p = preds[call_idx[0]]
        call_idx[0] += 1
        return p

    forecaster.predict = fake_predict
    result = forecaster.diagnose_error_accumulation(X_curr_sequence=[None, None, None], y_target_sequence=targets, accumulation_threshold=2.0)
    assert result["chain_mse"][0] == 0.0
    assert not np.isnan(result["growth_ratio"]).any()
    assert result["growth_ratio"][0] == 1.0
    assert result["trustworthy_horizon"] == 3


def test_f11_genuine_blowup_still_detected():
    """F11: genuine blowup still detected."""
    from sklearn.linear_model import LinearRegression

    from mlframe.training.composite.chained_window_forecast import ChainedWindowForecaster

    forecaster = ChainedWindowForecaster(stage1_estimator=LinearRegression(), stage2_estimator=LinearRegression())
    targets = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
    preds = [np.array([1.5]), np.array([1.6]), np.array([5.0])]  # nonzero baseline, then a real blowup
    call_idx = [0]

    def fake_predict(X):
        """Fake predict callable returning a fixed array."""
        p = preds[call_idx[0]]
        call_idx[0] += 1
        return p

    forecaster.predict = fake_predict
    result = forecaster.diagnose_error_accumulation(X_curr_sequence=[None, None, None], y_target_sequence=targets, accumulation_threshold=2.0)
    assert result["trustworthy_horizon"] == 2


# ---------------------------------------------------------------------------
# F12 (P2): additive_decomposition.py crashed with a cryptic torch error on component_names=()
# ---------------------------------------------------------------------------


def test_f12_empty_component_names_raises_clear_value_error():
    """F12: empty component names raises clear value error."""
    from mlframe.training.composite.additive_decomposition import AdditiveDecompositionRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    y = rng.normal(size=50).astype(np.float32)
    with pytest.raises(ValueError, match="non-empty"):
        AdditiveDecompositionRegressor(component_names=(), n_epochs=5).fit(X, y)


# ---------------------------------------------------------------------------
# F13 (P2, docs): _profile_pipeline.py documented a never-implemented --full flag
# ---------------------------------------------------------------------------


def test_f13_profile_pipeline_docstring_no_longer_mentions_full_flag():
    """F13: profile pipeline docstring no longer mentions full flag."""
    import mlframe.training.composite._profile_pipeline as pp_mod

    assert "--full" not in (pp_mod.__doc__ or "")


# ---------------------------------------------------------------------------
# F14 (P2, docs): venn_abers.py overstated the IVAP saddle kernel as "near-linear"
# ---------------------------------------------------------------------------


def test_f14_ivap_saddle_docstring_states_correct_complexity():
    """F14: ivap saddle docstring states correct complexity."""
    from mlframe.training.composite.venn_abers import _ivap_envelope

    doc = _ivap_envelope.__doc__ or ""
    assert "the kernel is near-linear" not in doc
    assert "O(g^2)" in doc
    assert "quadratically" in doc


# ---------------------------------------------------------------------------
# F15 (P2): multi_output.py's falsy-check silently overrode an explicit empty-string base_column
# ---------------------------------------------------------------------------


def test_f15_explicit_empty_string_base_column_not_overridden():
    """F15: explicit empty string base column not overridden."""
    from mlframe.training.composite.multi_output import CompositeMultiOutputEstimator

    est = CompositeMultiOutputEstimator(
        base_estimator=None,
        column_specs=[{"base_column": "", "transform_name": "diff"}, {"transform_name": "diff"}],
        base_columns_map={0: "should_not_override", 1: "fallback_base"},
    )
    specs = est._resolve_specs(n_outputs=2)
    assert specs[0]["base_column"] == ""
    assert specs[1]["base_column"] == "fallback_base"


# ---------------------------------------------------------------------------
# F16 (P2, test-gap): no test file existed for PerGroupCompositeRouter at all
# ---------------------------------------------------------------------------


def test_f16_per_group_router_basic_fit_predict():
    """F16: per group router basic fit predict."""
    router, X, _groups, _n_per_group = _fit_router()
    preds = router.predict(X)
    assert preds.shape == (len(X),)
    assert set(router.group_estimators_.keys()) == {"A", "B"}
    assert "C" not in router.group_estimators_  # group C had no per-group spec -> routes to global


# ---------------------------------------------------------------------------
# F17 (P2, test-gap): row_level_average_importance.py's helper functions were untested directly
# ---------------------------------------------------------------------------


def test_f17_extract_model_importance_tree_and_linear():
    """F17: extract model importance tree and linear."""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    from mlframe.training.composite.row_level_average_importance import extract_model_importance

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=200)

    tree_model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    imp_tree = extract_model_importance(tree_model, ["a", "b", "c"])
    assert imp_tree.shape == (3,)

    lin_model = LinearRegression().fit(X, y)
    imp_lin = extract_model_importance(lin_model, ["a", "b", "c"])
    assert imp_lin.shape == (3,)
    assert (imp_lin >= 0).all()  # abs(coef_)


def test_f17_extract_model_importance_unsupported_model_raises():
    """F17: extract model importance unsupported model raises."""
    from mlframe.training.composite.row_level_average_importance import extract_model_importance

    class NoImportanceModel:
        """Stub model exposing neither feature_importances_ nor coef_."""
        pass

    with pytest.raises(AttributeError):
        extract_model_importance(NoImportanceModel(), ["a"])


def test_f17_compute_row_level_feature_importance_oof_and_single():
    """F17: compute row level feature importance oof and single."""
    from sklearn.ensemble import RandomForestRegressor

    from mlframe.training.composite.row_level_average_importance import (
        compute_row_level_feature_importance_oof,
        compute_row_level_feature_importance_single_model,
    )

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = X["a"].to_numpy() + rng.normal(scale=0.1, size=n)
    entity = np.repeat(np.arange(30), 10)

    df_oof = compute_row_level_feature_importance_oof(
        lambda: RandomForestRegressor(n_estimators=10, random_state=0), X, y, entity, n_splits=3,
    )
    assert set(df_oof["feature"]) == {"a", "b"}
    assert df_oof["importance"][0] >= df_oof["importance"][1]  # sorted descending

    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    df_single = compute_row_level_feature_importance_single_model(model, X)
    assert set(df_single["feature"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# F18 (P2, test-gap): _booster_margin.py's family dispatch / error path had no dedicated test
# ---------------------------------------------------------------------------


def test_f18_booster_margin_family_dispatch_lgbm():
    """F18: booster margin family dispatch lgbm."""
    import lightgbm as lgb

    from mlframe.training.composite._booster_margin import inner_raw_margin

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.integers(0, 2, size=100)
    model = lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y)
    margin = inner_raw_margin(model, X, lgbm_attr="LGBMClassifier", xgb_attr="XGBClassifier", catboost_attr="CatBoostClassifier", wrapper_name="W", keep_2d=False)
    assert margin.shape == (100,)


def test_f18_booster_margin_dependency_not_installed_is_skipped_not_erroring():
    """A library genuinely not matching the target attr on an unrelated model type must fall through cleanly."""
    from mlframe.training.composite._booster_margin import inner_raw_margin

    class UnrelatedObject:
        """A plain object matching no supported booster family, for the dependency-not-installed skip path."""
        pass

    with pytest.raises(NotImplementedError):
        inner_raw_margin(UnrelatedObject(), None, lgbm_attr="LGBMClassifier", xgb_attr="XGBClassifier", catboost_attr="CatBoostClassifier", wrapper_name="W", keep_2d=False)
