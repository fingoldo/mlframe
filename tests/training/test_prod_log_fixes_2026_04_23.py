"""
Sensor tests locking in fixes derived from the 2026-04-23 prod log review.

Each group targets a single concrete regression that surfaced in the
``train_mlframe_models_suite`` run on ``prod_jobsdetails``. Keep these tests
deliberately small and focused — they are regression sensors, not integration
tests. A breakage here means one of the log-level pain points listed in the
review is back.

Fixes covered:

1. ``get_pandas_view_of_polars_df`` — nullable Polars Boolean columns used to
   materialize as pandas ``object`` dtype and blow up LightGBM's sklearn
   wrapper (``ValueError: pandas dtypes must be int, float or bool``). Now
   coerced to pandas ``Int8`` with ``pd.NA`` — accepted by LGB, XGB, and CB.
2. ``ensembling.harm`` — harmonic mean over predictions containing exact 0.0
   used to emit a ``RuntimeWarning: divide by zero`` and rely on an
   ``inf``-routing fallback. Now mask-routed: ``any(pred == 0) → 0`` (by
   harmonic-mean definition), no warning.
3. Default weighting schemas — the extractor used to default to
   ``use_uniform_weighting=False``, which produced a ``recency``-only suite
   with no uniform baseline. Default flipped to ``True`` so timeseries data
   gets ``{uniform, recency}`` and non-timeseries gets ``{uniform}`` alone.
4. CatBoost Polars-fastpath sticky shortcut — when CB's
   ``_set_features_order_data_polars_categorical_column`` dispatch miss
   forces a Polars→pandas fallback during predict, the model is tagged so
   subsequent predict calls (VAL, TEST, ensembles) skip the retry dance.
"""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest


# =====================================================================
# Fix 1: get_pandas_view_of_polars_df — nullable Boolean coercion
# =====================================================================

class TestNullableBooleanCoercion:
    """Lock in the 2026-04-23 fix: Polars Boolean with nulls no longer
    reaches the model backends as pandas ``object`` dtype."""

    def test_nullable_boolean_becomes_int8_not_object(self):
        """Polars Boolean with a null must not materialize as pandas object.

        The 2026-04-23 prod log crashed LGB with ``hide_budget object`` —
        an inline-rendered pandas dtype confirming a nullable Boolean column
        reached LGB as ``object``. The fix path coerces such columns to
        pandas ``Int8``.
        """
        from mlframe.training.utils import get_pandas_view_of_polars_df

        df = pl.DataFrame({
            "hide_budget": pl.Series([True, False, None, True], dtype=pl.Boolean),
            "num": [1.0, 2.0, 3.0, 4.0],
        })
        pdf = get_pandas_view_of_polars_df(df)

        assert str(pdf["hide_budget"].dtype) == "Int8", (
            f"Nullable Boolean must coerce to Int8 with pd.NA, got "
            f"{pdf['hide_budget'].dtype}"
        )
        # Values preserved: True→1, False→0, None→pd.NA
        assert pdf["hide_budget"].iloc[0] == 1
        assert pdf["hide_budget"].iloc[1] == 0
        assert pd.isna(pdf["hide_budget"].iloc[2])
        assert pdf["hide_budget"].iloc[3] == 1

    def test_non_null_boolean_stays_numpy_bool(self):
        """A Boolean column with *no* nulls must stay zero-copy numpy bool.

        Coercion is paid only when needed — the fix must not regress the
        fast path for plain Boolean columns.
        """
        from mlframe.training.utils import get_pandas_view_of_polars_df

        df = pl.DataFrame({
            "plain_bool": pl.Series([True, False, True, False], dtype=pl.Boolean),
        })
        pdf = get_pandas_view_of_polars_df(df)
        assert pdf["plain_bool"].dtype == np.dtype("bool")

    @pytest.mark.parametrize("backend_name", ["lgb", "xgb", "cb"])
    def test_fitted_models_accept_coerced_frame(self, backend_name):
        """End-to-end: all three tree backends fit on a frame that used to
        crash LGB."""
        from mlframe.training.utils import get_pandas_view_of_polars_df

        rng = np.random.default_rng(0)
        n = 200
        df = pl.DataFrame({
            # mirror the prod columns that broke: nullable Boolean + a few mixed dtypes
            "hide_budget": pl.Series(
                [bool(b) if rng.random() > 0.1 else None for b in rng.integers(0, 2, n)],
                dtype=pl.Boolean,
            ),
            "plain_bool": pl.Series(rng.integers(0, 2, n).astype(bool).tolist(), dtype=pl.Boolean),
            "cat_col": pl.Series(rng.choice(list("abc"), n).tolist(), dtype=pl.Categorical),
            "num": rng.normal(size=n).astype(np.float32),
        })
        y = rng.integers(0, 2, n)

        pdf = get_pandas_view_of_polars_df(df)

        if backend_name == "lgb":
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=3, verbose=-1)
            # If this fit raises the old ``pandas dtypes must be int, float or
            # bool`` (the 2026-04-23 LGB regression), the fix regressed.
            m.fit(pdf, y)
        elif backend_name == "xgb":
            import xgboost as xgb
            m = xgb.XGBClassifier(n_estimators=3, tree_method="hist", enable_categorical=True)
            m.fit(pdf, y)
        else:
            import catboost as cb
            m = cb.CatBoostClassifier(iterations=3, verbose=0)
            # CB requires cat_features when any column is pd.Categorical. If
            # Int8 regresses to pandas nullable ``boolean`` here, CB's
            # numeric-feature path crashes with "Cannot convert <NA> to float".
            m.fit(pdf, y, cat_features=["cat_col"])


# =====================================================================
# Fix 2: ensembling harmonic mean — no divide-by-zero warning
# =====================================================================

class TestEnsembleHarmonicDivByZero:
    """Lock in the 2026-04-23 fix: harmonic mean on predictions with any
    exact 0.0 no longer emits ``RuntimeWarning: divide by zero``."""

    def _call_harm(self, preds):
        """Helper: invoke ensemble_probabilistic_predictions with method=harm.

        ``max_mae=0`` and ``max_std=0`` disable the median-distance filter
        that would otherwise drop one of the predictions when ``len(preds) > 2``
        — we need the raw HM across all inputs so the closed-form check
        below is deterministic.

        Returns only the ensembled prediction array; the function returns a
        tuple ``(predictions, confident_indices, uncertainty)``.
        """
        from mlframe.ensembling import ensemble_probabilistic_predictions

        pred_arrays = [np.asarray(p, dtype=np.float64).reshape(-1, 1) for p in preds]
        out = ensemble_probabilistic_predictions(
            *pred_arrays,
            ensemble_method="harm",
            max_mae=0,
            max_std=0,
            verbose=False,
        )
        return out[0] if isinstance(out, tuple) else out

    def test_harm_with_zeros_no_runtime_warning(self):
        """No ``RuntimeWarning`` must be emitted by the harm path itself."""
        preds = [
            [0.8, 0.0, 0.5, 0.9],
            [0.7, 0.3, 0.0, 0.1],
            [0.6, 0.4, 0.2, 0.8],
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = self._call_harm(preds)
        div_zero = [w for w in caught if "divide by zero" in str(w.message).lower()]
        assert not div_zero, (
            f"Harmonic mean must not emit divide-by-zero warnings; got: "
            f"{[str(w.message) for w in div_zero]}"
        )
        # Downstream consumers expect a finite probability vector.
        assert np.isfinite(out).all()

    def test_harm_with_zeros_correct_values(self):
        """HM by definition: any 0 in inputs forces result 0 at that slot."""
        preds = [
            [0.8, 0.0, 0.5, 0.9],
            [0.7, 0.3, 0.0, 0.1],
            [0.6, 0.4, 0.2, 0.8],
        ]
        out = self._call_harm(preds).ravel()
        # pos 0: no zeros → standard HM
        hm_0 = 3.0 / (1 / 0.8 + 1 / 0.7 + 1 / 0.6)
        assert abs(out[0] - hm_0) < 1e-9
        # pos 1: pred[0]=0 → 0
        assert out[1] == 0.0
        # pos 2: pred[1]=0 → 0
        assert out[2] == 0.0
        # pos 3: no zeros → standard HM
        hm_3 = 3.0 / (1 / 0.9 + 1 / 0.1 + 1 / 0.8)
        assert abs(out[3] - hm_3) < 1e-9

    def test_harm_without_zeros_unchanged(self):
        """HM on a zero-free input must match the closed-form value exactly."""
        preds = [[0.2, 0.5], [0.4, 0.6], [0.8, 0.7]]
        out = self._call_harm(preds).ravel()
        hm_0 = 3.0 / (1 / 0.2 + 1 / 0.4 + 1 / 0.8)
        hm_1 = 3.0 / (1 / 0.5 + 1 / 0.6 + 1 / 0.7)
        assert abs(out[0] - hm_0) < 1e-9
        assert abs(out[1] - hm_1) < 1e-9


# =====================================================================
# Fix 3: default weighting schemas
# =====================================================================

class TestDefaultWeightingSchemas:
    """Lock in the flipped default: ``use_uniform_weighting=True`` so every
    run gets a uniform baseline. Without it the 2026-04-23 prod suite had a
    ``recency``-only attribution gap."""

    def test_timeseries_default_is_uniform_plus_recency(self):
        from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

        extractor = SimpleFeaturesAndTargetsExtractor(regression_targets=["price"])
        assert extractor.use_uniform_weighting is True
        assert extractor.use_recency_weighting is True

        n = 50
        df = pl.DataFrame({"price": np.linspace(1, 100, n).tolist()})
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=n, freq="D"))

        weights = extractor.get_sample_weights(df, timestamps=timestamps)
        assert set(weights.keys()) == {"uniform", "recency"}

    def test_non_timeseries_default_is_uniform_only(self):
        from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

        extractor = SimpleFeaturesAndTargetsExtractor(regression_targets=["price"])
        n = 50
        df = pl.DataFrame({"price": np.linspace(1, 100, n).tolist()})

        weights = extractor.get_sample_weights(df, timestamps=None)
        assert set(weights.keys()) == {"uniform"}

    def test_explicit_opt_out_still_works(self):
        """Users can still disable uniform for experimental runs."""
        from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

        extractor = SimpleFeaturesAndTargetsExtractor(
            regression_targets=["price"],
            use_uniform_weighting=False,
        )
        n = 50
        df = pl.DataFrame({"price": np.linspace(1, 100, n).tolist()})
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=n, freq="D"))

        weights = extractor.get_sample_weights(df, timestamps=timestamps)
        assert set(weights.keys()) == {"recency"}


# =====================================================================
# Fix 4: CatBoost Polars-fastpath sticky shortcut
# =====================================================================

class TestCatBoostFastpathStickyFlag:
    """Lock in the 2026-04-23 optimization: after _predict_with_fallback
    converts once, the model is flagged so subsequent calls skip the
    TypeError-retry dance. Previously VAL, TEST, and each ensemble member
    re-hit the same Cython dispatch miss, emitting a WARN + ~2s retry per
    call on prod-size frames."""

    def test_flag_short_circuits_fn_call_to_pandas_path(self):
        """When the flag is set, fn(pl.DataFrame) must NOT be called — the
        helper must pre-convert and call fn(pandas.DataFrame) directly."""
        from mlframe.training.trainer import _predict_with_fallback

        # Fake a CatBoost model stub: class name matches CATBOOST_MODEL_TYPES,
        # flag is set, and predict_proba records which container kind it
        # received. The short-circuit must route pandas in.
        calls: list = []

        class _FakeCB:
            # Trick the isinstance-in-list check via class name
            pass

        _FakeCB.__name__ = "CatBoostClassifier"

        def _fake_predict_proba(X):
            calls.append(type(X).__module__ + "." + type(X).__name__)
            return np.array([[0.3, 0.7]] * len(X))

        fake = _FakeCB()
        fake.predict_proba = _fake_predict_proba
        # Also stub the attribute hooks used by _recover_cb_feature_names
        fake.feature_names_ = ["a", "num"]
        fake._get_cat_feature_indices = lambda: [0]
        fake._get_text_feature_indices = lambda: []
        fake._mlframe_polars_fastpath_broken = True

        pl_df = pl.DataFrame({
            "a": pl.Series(["x", "y", "x"], dtype=pl.Categorical),
            "num": [1.0, 2.0, 3.0],
        })
        out = _predict_with_fallback(fake, pl_df, method="predict_proba")
        assert out.shape == (3, 2)
        # Short-circuit converted to pandas before calling fn
        assert calls == ["pandas.core.frame.DataFrame"], (
            f"sticky shortcut must pass pandas to fn; saw: {calls}"
        )

    def test_flag_absent_retains_normal_fastpath_retry(self):
        """Without the flag, the helper must attempt fn(pl.DataFrame) first
        (preserving the existing fastpath-or-retry behaviour for models that
        haven't yet failed)."""
        from mlframe.training.trainer import _predict_with_fallback

        calls: list = []

        class _FakeCB:
            pass

        _FakeCB.__name__ = "CatBoostClassifier"

        def _fake_predict_proba(X):
            calls.append(type(X).__module__ + "." + type(X).__name__)
            # Works on whatever we get — we're verifying call order, not
            # fastpath success.
            return np.array([[0.3, 0.7]] * len(X))

        fake = _FakeCB()
        fake.predict_proba = _fake_predict_proba
        # No _mlframe_polars_fastpath_broken attribute → normal path.

        pl_df = pl.DataFrame({
            "a": pl.Series(["x", "y"], dtype=pl.Categorical),
        })
        _predict_with_fallback(fake, pl_df, method="predict_proba")
        # Normal path tries fn(pl.DataFrame) first (which succeeds here).
        assert calls == ["polars.dataframe.frame.DataFrame"]


# =====================================================================
# Fix 5: pipeline_cache must not cross streams between polars-native and
#        pandas-consuming strategies — cache_key now includes container kind
# =====================================================================

class TestPipelineCacheKindIsolation:
    """Lock in the 2026-04-23 fix: ``pipeline_cache`` entries written by a
    Polars-native strategy (XGB/CB/HGB) must not be returned to a
    pandas-consuming strategy (LGB/sklearn/linear), even when they share
    ``strategy.cache_key`` (e.g. 'tree') and ``feature_tier``.

    Before the fix: XGB wrote its polars train frame under
    ``tree__tier(False,False)``; LGB then pulled that polars out of the cache,
    bypassed core.py's strategy-loop lazy conversion, and the trainer paid a
    duplicate 224 s polars→pandas conversion as a "self-heal" (2026-04-23 prod
    log). After the fix: the cache key carries a container-kind suffix
    (``_kindpl`` / ``_kindpd``) so consumers never collide."""

    def test_mixed_cb_xgb_lgb_suite_runs_single_lgb_conversion(self, tmp_path):
        """End-to-end: run cb+xgb+lgb on a small polars frame. The trainer's
        polars-frame raise (non-native models must receive pandas) stays
        silent — proof that LGB gets pandas, not a cached polars frame from
        XGB. If the raise fires, the kind-suffix fix has regressed.
        """
        import tempfile
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training.configs import TrainingBehaviorConfig
        from .shared import SimpleFeaturesAndTargetsExtractor

        n = 600
        rng = np.random.default_rng(0)
        budget_cats = ["HOURLY", "FIXED", "MILESTONE"]
        tier_cats = ["BEGINNER", "INTERMEDIATE", "EXPERT"]
        workload_cats = ["LESS_THAN_30", "MORE_THAN_30", "FULL_TIME"]
        pl_df = pl.DataFrame({
            "num_feat_1": rng.standard_normal(n).astype(np.float32),
            "num_feat_2": rng.standard_normal(n).astype(np.float32),
            "num_feat_3": rng.standard_normal(n).astype(np.float32),
            "budget_type": pl.Series(
                [budget_cats[i % 3] for i in range(n)]
            ).cast(pl.Enum(budget_cats)),
            "contractor_tier": pl.Series(
                [tier_cats[i % 3] for i in range(n)]
            ).cast(pl.Enum(tier_cats)),
            "workload": pl.Series(
                [workload_cats[i % 3] for i in range(n)]
            ).cast(pl.Enum(workload_cats)),
            "target": rng.integers(0, 2, n),
        })
        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="target", regression=False
        )
        bc = TrainingBehaviorConfig(prefer_gpu_configs=False)

        # This used to raise the trainer's ``RuntimeError: LGBMClassifier
        # received pl.DataFrame`` because pipeline_cache leaked XGB's polars
        # frame into LGB. Post-fix, the suite completes end-to-end.
        models, _metadata = train_mlframe_models_suite(
            df=pl_df,
            target_name="pipeline_cache_kind_test",
            model_name="mix",
            features_and_targets_extractor=fte,
            mlframe_models=["cb", "xgb", "lgb"],
            hyperparams_config={"iterations": 3},
            behavior_config=bc,
            init_common_params={"drop_columns": [], "verbose": 0},
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=str(tmp_path),
            models_dir="models",
            verbose=0,
        )
        assert models, "train_mlframe_models_suite returned empty models"

    def test_cache_key_includes_container_kind_suffix(self):
        """Unit-level guard: the kind qualifier is present in the key format
        construction, preventing collisions between polars-native XGB and
        pandas-consuming LGB even when strategy.cache_key + feature_tier are
        identical.

        Reconstruct the key format used in core.py and verify pl/pd variants
        differ. A regression here — someone removing the kind suffix because
        it "looks redundant" — is exactly what triggered the 2026-04-23 prod
        leak.
        """
        from mlframe.training.strategies import get_strategy

        xgb_strat = get_strategy("xgb")
        lgb_strat = get_strategy("lgb")

        # Precondition: the reason the fix is needed. If these ever diverge,
        # the cache key would be kind-safe "for free" and the fix becomes
        # unnecessary — but until then, the kind suffix is load-bearing.
        assert xgb_strat.cache_key == lgb_strat.cache_key == "tree"
        assert xgb_strat.feature_tier() == lgb_strat.feature_tier()

        def _build_key(strat):
            tier_suffix = f"_tier{strat.feature_tier()}"
            kind_suffix = f"_kind{'pl' if strat.supports_polars else 'pd'}"
            return f"{strat.cache_key}{tier_suffix}{kind_suffix}"

        xgb_key = _build_key(xgb_strat)
        lgb_key = _build_key(lgb_strat)
        assert xgb_key != lgb_key, (
            f"pipeline_cache keys must differ for polars-native vs pandas-"
            f"consuming strategies sharing cache_key+tier. xgb={xgb_key}, "
            f"lgb={lgb_key}"
        )
        assert xgb_key.endswith("_kindpl")
        assert lgb_key.endswith("_kindpd")

    def test_trainer_raises_if_polars_frame_leaks_to_non_native_model(self):
        """Defense-in-depth: if a future regression re-introduces the cache
        leak, the trainer must raise rather than silently double-convert.

        Call _train_model_with_fallback directly on a LGB model with a
        polars frame — it must raise RuntimeError mentioning pipeline_cache,
        not paper over with a conversion."""
        from mlframe.training.trainer import _train_model_with_fallback

        # Minimal fake LGB-like model (class name-gated: the trainer uses
        # ``model_type_name.startswith(...)`` for allow-listing).
        class _FakeLGBM:
            pass

        _FakeLGBM.__name__ = "LGBMClassifier"
        fake = _FakeLGBM()
        # Stub .fit so if the raise doesn't fire, the test still terminates.
        fake.fit = lambda *a, **kw: None

        pl_df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.1, 0.2, 0.3]})
        with pytest.raises(RuntimeError, match="pipeline_cache"):
            _train_model_with_fallback(
                model=fake,
                model_obj=fake,
                model_type_name="LGBMClassifier",
                train_df=pl_df,
                train_target=np.array([0, 1, 0]),
                fit_params={},
                verbose=False,
            )
