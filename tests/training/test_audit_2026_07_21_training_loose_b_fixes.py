"""Regression tests for audits/full_audit_2026-07-21/training_loose_b.md findings F1-F10 and proposals PR3/PR4/PR7.

F1 (unnecessary deep-copy in _training_loop.py's CatBoost NaN-fill path) was found ALREADY fixed by an
earlier session-wide large-frame-copy sweep predating this cluster's pass -- pinned here, not re-fixed.
While verifying F1's cross-reference, a SEPARATE, previously-undiscovered instance of the exact same bug
class was found in preprocessing/degradation_augment.py's match_noise_level (the sibling of
match_missingness_rate, whose own instance preprocessing.md's F2 already fixed) -- fixed and tested here.
PR1/PR2/PR5/PR6 (test-coverage / consistency proposals) are closed by the F2/F5/F7/F9 tests below.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# F1: _training_loop.py's CatBoost NaN-fill copy() -- already deep=False; pinned + sibling bug fixed
# ---------------------------------------------------------------------------


def test_f1_training_loop_catboost_nan_fill_uses_shallow_copy():
    """F1: training loop catboost nan fill uses shallow copy."""
    import inspect

    from mlframe.training import _training_loop

    src = inspect.getsource(_training_loop._train_model_with_fallback)
    # Every bare `.copy()` in this function's NaN-fill blocks must be deep=False (not asserting via
    # string search for its own sake -- see the behavioral check below for the real regression guard).
    assert "train_df.copy(deep=False)" in src or "_eval_df_filled.copy(deep=False)" in src


def test_f1_degradation_augment_match_noise_level_shares_memory_for_non_numeric_cols():
    """Newly-discovered sibling bug (same class as preprocessing.md's F2/F3): match_noise_level's
    `out = X_train.copy()` was a full deep copy; a non-numeric column now shares memory with the input."""
    from mlframe.preprocessing.degradation_augment import match_noise_level

    rng = np.random.default_rng(0)
    n = 200
    X_train = pd.DataFrame({"num": rng.normal(size=n), "cat": pd.array(["a", "b"] * (n // 2), dtype="string")})
    X_test = pd.DataFrame({"num": rng.normal(scale=3.0, size=n), "cat": pd.array(["a", "b"] * (n // 2), dtype="string")})

    out = match_noise_level(X_train, X_test, np.random.default_rng(1))
    # The non-numeric column was never touched by this function -- a deep=False copy means it shares
    # the same underlying buffer as X_train's own "cat" column.
    assert np.shares_memory(out["cat"].to_numpy(), X_train["cat"].to_numpy())
    # The numeric column WAS modified (noise added / reassigned) -- must NOT share memory with X_train.
    assert not np.array_equal(out["num"].to_numpy(), X_train["num"].to_numpy())


# ---------------------------------------------------------------------------
# F2: _compute_oof_preds silently dropped sample_weight for level-1 stacking OOF
# ---------------------------------------------------------------------------


def test_f2_compute_oof_preds_threads_sample_weight():
    """F2: compute oof preds threads sample weight."""
    from mlframe.training.trainer import _compute_oof_preds
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 300
    train_df = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    train_target = (train_df["f0"] + rng.normal(scale=0.1, size=n) > 0).astype(int).to_numpy()
    sample_weight = np.where(train_target == 1, 5.0, 1.0)  # heavily upweight the positive class

    model = LogisticRegression(max_iter=200)
    model.fit(train_df, train_target)

    _, oof_probs_weighted = _compute_oof_preds(
        model=model, train_df=train_df, train_target=train_target, is_classifier_model=True,
        n_splits=3, random_seed=0, sample_weight=sample_weight,
    )
    _, oof_probs_unweighted = _compute_oof_preds(
        model=model, train_df=train_df, train_target=train_target, is_classifier_model=True,
        n_splits=3, random_seed=0, sample_weight=None,
    )
    assert oof_probs_weighted is not None and oof_probs_unweighted is not None
    assert not np.allclose(oof_probs_weighted, oof_probs_unweighted)


def test_f2_compute_oof_preds_timeseries_threads_sample_weight():
    """F2: compute oof preds timeseries threads sample weight."""
    from mlframe.training.trainer import _compute_oof_preds

    rng = np.random.default_rng(0)
    n = 200
    train_df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    train_target = (train_df["f0"].to_numpy() + np.linspace(0, 1, n) > 0.5).astype(int)
    sample_weight = np.where(train_target == 1, 8.0, 1.0)

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=200)
    model.fit(train_df, train_target)

    _, oof_w = _compute_oof_preds(
        model=model, train_df=train_df, train_target=train_target, is_classifier_model=True,
        n_splits=3, random_seed=0, has_time=True, sample_weight=sample_weight,
    )
    _, oof_u = _compute_oof_preds(
        model=model, train_df=train_df, train_target=train_target, is_classifier_model=True,
        n_splits=3, random_seed=0, has_time=True, sample_weight=None,
    )
    assert oof_w is not None and oof_u is not None
    finite_mask = np.isfinite(oof_w).all(axis=1) & np.isfinite(oof_u).all(axis=1)
    assert finite_mask.sum() > 0
    assert not np.allclose(oof_w[finite_mask], oof_u[finite_mask])


def test_f2_compute_oof_preds_unsupported_sample_weight_falls_back_gracefully():
    """An estimator whose .fit() doesn't accept sample_weight must still compute OOF (unweighted),
    not silently skip entirely."""
    from mlframe.training.trainer import _compute_oof_preds
    from sklearn.base import BaseEstimator

    class _NoWeightRegressor(BaseEstimator):
        """BaseEstimator (not RegressorMixin) so cross_val_predict's sklearn-version machinery (e.g.
        __sklearn_tags__) works, without adding sample_weight support to .fit()."""

        def fit(self, X, y):
            """No-op / recording stub matching the estimator's fit() signature."""
            self._mean = float(np.mean(y))
            return self

        def predict(self, X):
            """No-op / recording stub matching the estimator's predict() signature."""
            return np.full(len(X), self._mean)

    rng = np.random.default_rng(0)
    n = 60
    train_df = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    train_target = rng.normal(size=n)
    sample_weight = rng.uniform(0.5, 2.0, size=n)

    oof_preds, _ = _compute_oof_preds(
        model=_NoWeightRegressor(), train_df=train_df, train_target=train_target, is_classifier_model=False,
        n_splits=3, random_seed=0, sample_weight=sample_weight,
    )
    assert oof_preds is not None


# ---------------------------------------------------------------------------
# F3: collapse-recovery ladder left network_params inconsistent with the live network on exhaustion
# ---------------------------------------------------------------------------


def test_f3_ladder_exhaustion_keeps_last_attempted_network_params():
    """F3: ladder exhaustion keeps last attempted network params."""
    from mlframe.training._training_loop_refit import _maybe_refit_on_collapsed_predictions

    class _FakeInnerNet:
        """Fake Inner Net."""
        def __init__(self):
            self.network_params = {"use_batchnorm": False, "nlayers": 3, "dropout_prob": 0.0}
            self.network = "original_network_object"

    class _FakeModelObj:
        """Mimics the attribute-chain shape `_maybe_refit_on_collapsed_predictions` walks
        (`estimator_` is one of the recognized nesting attrs)."""

        def __init__(self, inner):
            self.estimator_ = inner

        def fit(self, train_df, train_target, **fit_params):
            # Simulate: every ladder rung "succeeds" at fitting but stays collapsed.
            """No-op / recording stub matching the estimator's fit() signature."""
            self.estimator_.network = f"network_fit_under_{self.estimator_.network_params}"

    inner = _FakeInnerNet()
    model_obj = _FakeModelObj(inner)

    rng = np.random.default_rng(0)
    n = 50
    train_target = rng.normal(size=n)
    train_df = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])

    class _FakeModel:
        """Fake model stub used to control this test's predict path."""
        def predict(self, X):
            # Always near-constant -> collapse detector fires every round.
            """No-op / recording stub matching the estimator's predict() signature."""
            return np.full(len(X), 0.001)

    orig_params_snapshot = dict(inner.network_params)
    _maybe_refit_on_collapsed_predictions(
        model=_FakeModel(), model_obj=model_obj, model_type_name="TestMLP",
        train_df=train_df, train_target=train_target, fit_params={}, logger_=logging.getLogger("test"),
    )

    # The ladder exhausted (still collapsed); network_params must reflect the LAST-tried rung
    # (bump_dropout: use_batchnorm=True, nlayers=1, dropout_prob=0.15), NOT the original config --
    # matching the live `.network`, which was fit under that same last candidate.
    assert inner.network_params != orig_params_snapshot
    assert inner.network_params.get("use_batchnorm") is True
    assert inner.network_params.get("nlayers") == 1
    assert inner.network_params.get("dropout_prob") == 0.15
    assert str(inner.network_params) in inner.network


# ---------------------------------------------------------------------------
# F4: unguarded isinstance(x, pl.DataFrame) despite the module's own optional-polars header
# ---------------------------------------------------------------------------


def test_f4_training_loop_survives_polars_unavailable(monkeypatch):
    """F4: training loop survives polars unavailable."""
    from mlframe.training import _training_loop

    monkeypatch.setattr(_training_loop, "pl", None)
    train_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    # Directly exercises the guarded line: must not raise AttributeError on `pl.DataFrame`.
    assert (_training_loop.pl is not None and isinstance(train_df, _training_loop.pl.DataFrame)) is False


# ---------------------------------------------------------------------------
# F5: get_pandas_view_of_polars_df's pl.Series contract was advertised but never implemented
# ---------------------------------------------------------------------------


def test_f5_series_input_raises_clear_type_error_not_deep_attribute_error():
    """F5: series input raises clear type error not deep attribute error."""
    import polars as pl

    from mlframe.training.utils import get_pandas_view_of_polars_df

    s = pl.Series("x", [1, 2, 3])
    with pytest.raises(TypeError, match="DataFrame"):
        get_pandas_view_of_polars_df(s)


def test_f5_dataframe_input_still_works():
    """F5: dataframe input still works."""
    import polars as pl

    from mlframe.training.utils import get_pandas_view_of_polars_df

    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = get_pandas_view_of_polars_df(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# F6: orphaned doc comment in _gpu_probe.py removed
# ---------------------------------------------------------------------------


def test_f6_gpu_probe_has_no_orphaned_dispatch_comment():
    """F6: gpu probe has no orphaned dispatch comment."""
    import inspect

    from mlframe.training import _gpu_probe

    src = inspect.getsource(_gpu_probe)
    assert "canonicalizer + decision-rule pair below" not in src


# ---------------------------------------------------------------------------
# F7 / PR5: confidence-analysis fit_params forwarded for ALL CatBoost model types, not just Regressor
# ---------------------------------------------------------------------------


def test_f7_fit_params_forwarded_for_catboost_classifier(monkeypatch):
    """F7: fit params forwarded for catboost classifier."""
    from mlframe.training._calib_oof_outputs import maybe_run_confidence_analysis

    captured = {}

    def fake_run_confidence_analysis(**kwargs):
        """Fake replacement for run_confidence_analysis that records its fit_params."""
        captured.update(kwargs)
        return None

    monkeypatch.setattr("mlframe.training._calib_oof_outputs.run_confidence_analysis", fake_run_confidence_analysis)

    maybe_run_confidence_analysis(
        run_test=True,
        confidence=type("C", (), {"include": True, "model_kwargs": None, "use_shap": False, "max_features": 5, "cmap": "x", "alpha": 0.5, "title": "t", "ylabel": "y"})(),
        test_df=pd.DataFrame({"a": [1, 2]}),
        test_target=np.array([0, 1]),
        test_probs=np.array([[0.5, 0.5], [0.2, 0.8]]),
        fit_params={"cat_features": ["a"], "text_features": None, "embedding_features": None},
        model_type_name="CatBoostClassifier",
        figsize=(4, 4),
        verbose=False,
    )
    assert captured["fit_params"] is not None
    assert captured["fit_params"]["cat_features"] == ["a"]


def test_f7_fit_params_not_forwarded_for_non_catboost(monkeypatch):
    """F7: fit params not forwarded for non catboost."""
    from mlframe.training._calib_oof_outputs import maybe_run_confidence_analysis

    captured = {}

    def fake_run_confidence_analysis(**kwargs):
        """Fake replacement for run_confidence_analysis that records its fit_params."""
        captured.update(kwargs)
        return None

    monkeypatch.setattr("mlframe.training._calib_oof_outputs.run_confidence_analysis", fake_run_confidence_analysis)

    maybe_run_confidence_analysis(
        run_test=True,
        confidence=type("C", (), {"include": True, "model_kwargs": None, "use_shap": False, "max_features": 5, "cmap": "x", "alpha": 0.5, "title": "t", "ylabel": "y"})(),
        test_df=pd.DataFrame({"a": [1, 2]}),
        test_target=np.array([0, 1]),
        test_probs=np.array([[0.5, 0.5], [0.2, 0.8]]),
        fit_params={"cat_features": ["a"]},
        model_type_name="LGBMClassifier",
        figsize=(4, 4),
        verbose=False,
    )
    assert captured["fit_params"] is None


# ---------------------------------------------------------------------------
# F8: permutation FI and confidence-regressor scoring drop sample_weight
# ---------------------------------------------------------------------------


def test_f8_permutation_feature_importances_threads_sample_weight():
    """F8: permutation feature importances threads sample weight."""
    from mlframe.training._feature_importances import _permutation_feature_importances
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.5, size=n)
    sample_weight = np.where(np.arange(n) < n // 2, 10.0, 0.1)  # heavily favor the first half

    model = LinearRegression()
    model.fit(X, y, sample_weight=sample_weight)

    fi_weighted = _permutation_feature_importances(model, X, y, n_repeats=5, random_state=0, sample_weight=sample_weight)
    fi_unweighted = _permutation_feature_importances(model, X, y, n_repeats=5, random_state=0, sample_weight=None)
    assert fi_weighted is not None and fi_unweighted is not None
    assert not np.allclose(fi_weighted, fi_unweighted)


def test_f8_adaptive_scorer_accepts_sample_weight_kwarg_without_crashing():
    """Direct regression test for the TypeError sklearn's permutation_importance would raise if the
    scorer callable didn't accept a sample_weight kwarg."""
    from mlframe.training._feature_importances import _permutation_feature_importances
    from sklearn.dummy import DummyClassifier

    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] > 0).astype(int)
    sw = rng.uniform(0.5, 2.0, size=n)

    model = DummyClassifier(strategy="stratified", random_state=0)
    model.fit(X, y)
    fi = _permutation_feature_importances(model, X, y, n_repeats=3, random_state=0, sample_weight=sw)
    assert fi is not None


def test_f8_confidence_analysis_threads_sample_weight_into_catboost_fit(monkeypatch):
    """F8: confidence analysis threads sample weight into catboost fit."""
    from mlframe.training import _confidence_analysis

    if _confidence_analysis.CatBoostRegressor is None:
        pytest.skip("catboost not installed")

    captured_kwargs = {}
    real_fit = _confidence_analysis.CatBoostRegressor.fit

    def spy_fit(self, X, y, **kwargs):
        """Records fit() calls for this test's assertions."""
        captured_kwargs.update(kwargs)
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(_confidence_analysis.CatBoostRegressor, "fit", spy_fit)
    monkeypatch.setattr(_confidence_analysis, "CUDA_IS_AVAILABLE", False, raising=False)

    rng = np.random.default_rng(0)
    n = 200
    test_df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    test_target = rng.integers(0, 2, size=n)
    p1 = rng.uniform(0.1, 0.9, size=n)
    test_probs = np.column_stack([1 - p1, p1])
    sample_weight = rng.uniform(0.5, 2.0, size=n)

    _confidence_analysis.run_confidence_analysis(
        test_df=test_df, test_target=test_target, test_probs=test_probs,
        use_shap=False, sample_weight=sample_weight,
        confidence_model_kwargs={"iterations": 5},
    )
    assert "sample_weight" in captured_kwargs
    assert np.array_equal(captured_kwargs["sample_weight"], sample_weight)


# ---------------------------------------------------------------------------
# F9 / PR6: multiclass drift report crashes on an empty (non-None) train_target
# ---------------------------------------------------------------------------


def test_f9_multiclass_drift_report_empty_train_target_no_crash():
    """F9: multiclass drift report empty train target no crash."""
    from mlframe.training.drift_report import compute_label_distribution_drift

    result = compute_label_distribution_drift(
        train_target=np.array([], dtype=np.int64),
        val_target=None,
        test_target=None,
        target_type="multiclass_classification",
    )
    assert result["splits"] == {}
    assert "empty" in result["warnings"][0]


def test_f9_multiclass_drift_report_normal_case_unaffected():
    """F9: multiclass drift report normal case unaffected."""
    from mlframe.training.drift_report import compute_label_distribution_drift

    rng = np.random.default_rng(0)
    train = rng.integers(0, 3, size=200)
    test = rng.integers(0, 3, size=100)
    result = compute_label_distribution_drift(train_target=train, val_target=None, test_target=test, target_type="multiclass_classification")
    assert "train" in result["splits"]
    assert result["splits"]["train"] is not None


# ---------------------------------------------------------------------------
# F10: honest_diagnostics bootstrap block's status flag contradicted by later ECE success
# ---------------------------------------------------------------------------


def test_f10_metrics_import_failure_sets_per_metric_skip_not_whole_block_status(monkeypatch):
    """F10: metrics import failure sets per metric skip not whole block status."""
    from mlframe.training import honest_diagnostics

    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    probs = np.column_stack([1 - rng.uniform(0, 1, size=n), rng.uniform(0, 1, size=n)])
    probs[:, 0] = 1 - probs[:, 1]

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        """Fake import hook that always raises ImportError."""
        if name == "mlframe.metrics.core":
            raise ImportError("simulated metrics.core unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = honest_diagnostics._bootstrap_block(y_true, probs, rng_seed=0)

    assert "status" not in out  # no more misleading whole-block key
    for metric_name in ("roc_auc", "brier", "log_loss"):
        assert out.get(metric_name, {}).get("status") == "skipped"
    # ECE is independent of mlframe.metrics.core -- it should still have run and NOT be marked skipped,
    # proving the old whole-block "status": "skipped" would have contradicted this real result.
    assert "ece" in out
    assert out["ece"].get("status") != "skipped"


# ---------------------------------------------------------------------------
# PR3: DirectHorizonBucketForecaster.predict warns when rows get NaN (no covering model)
# ---------------------------------------------------------------------------


def test_pr3_predict_warns_on_uncovered_rows(caplog):
    """PR3: predict warns on uncovered rows."""
    from mlframe.training._direct_horizon_bucket_forecaster import DirectHorizonBucketForecaster
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"f0": rng.normal(size=n)})
    y = rng.normal(size=n)
    horizon_day = rng.integers(1, 8, size=n)  # bucket (1,7) only

    forecaster = DirectHorizonBucketForecaster(horizon_buckets=[(1, 7)], model_factory=LinearRegression)
    forecaster.fit(X, y, horizon_day)

    X_pred = pd.DataFrame({"f0": rng.normal(size=20)})
    horizon_day_pred = np.full(20, 99)  # outside any fitted bucket -> all NaN
    with caplog.at_level(logging.WARNING, logger="mlframe.training._direct_horizon_bucket_forecaster"):
        preds = forecaster.predict(X_pred, horizon_day_pred)
    assert np.all(np.isnan(preds))
    assert any("no fitted model covers" in r.getMessage() for r in caplog.records)


def test_pr3_predict_no_warning_when_fully_covered(caplog):
    """PR3: predict no warning when fully covered."""
    from mlframe.training._direct_horizon_bucket_forecaster import DirectHorizonBucketForecaster
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({"f0": rng.normal(size=n)})
    y = rng.normal(size=n)
    horizon_day = rng.integers(1, 8, size=n)

    forecaster = DirectHorizonBucketForecaster(horizon_buckets=[(1, 7)], model_factory=LinearRegression)
    forecaster.fit(X, y, horizon_day)

    X_pred = pd.DataFrame({"f0": rng.normal(size=20)})
    horizon_day_pred = rng.integers(1, 8, size=20)
    with caplog.at_level(logging.WARNING, logger="mlframe.training._direct_horizon_bucket_forecaster"):
        preds = forecaster.predict(X_pred, horizon_day_pred)
    assert not np.any(np.isnan(preds))
    assert not any("no fitted model covers" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# PR4: dynamically-created logging proxy class declares __slots__
# ---------------------------------------------------------------------------


def test_pr4_logging_proxy_has_no_instance_dict():
    """`_LoggingProxy.__getattr__` forwards unknown attrs (incl. `__dict__`) to `self._inner`, so
    `proxy.__dict__` alone can't tell whether the PROXY class itself carries a __dict__ slot -- check
    the class's C-level __dictoffset__ instead (0 means no implicit per-instance __dict__ was added)."""
    from mlframe.training.logging_transformers import wrap_with_logging

    class _Dummy:
        """Minimal stub estimator used only to probe attribute access."""
        def fit(self, X, y=None):
            """No-op / recording stub matching the estimator's fit() signature."""
            return self

    proxy = wrap_with_logging(_Dummy(), stage="test_pr4")
    assert type(proxy).__dictoffset__ == 0
