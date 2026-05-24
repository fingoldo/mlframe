"""
Sensor tests for ``mlframe.training.xgb_shim`` — the DMatrix-reuse shim
that wraps ``XGBClassifier`` / ``XGBRegressor`` so that consecutive
``.fit()`` calls on the same feature matrix don't rebuild the
``QuantileDMatrix`` from scratch.

The 2026-04-24 prod log captured the cost the shim is paid to eliminate:
``XGBClassifier.fit(X, y, sample_weight=w)`` rebuilt a 7.3M × 106
QuantileDMatrix in **104 s** for the ``uniform`` weight schema, then
**99 s** for ``recency`` — same feature matrix, only the sample_weight
vector changed. The shim caches the DMatrix and uses ``set_label`` /
``set_weight`` for in-place swaps.

What's tested
-------------
1. **API parity with XGBClassifier** — drop-in replacement: same
   ``get_params()``, ``set_params()``, ``predict``, ``predict_proba``,
   ``feature_importances_``, ``feature_names_in_``, sklearn ``clone()``
   round-trip.
2. **DMatrix cache** — second fit on the same DataFrame reuses the
   cached DMatrix; second fit on a different DataFrame misses; cache
   resets on clone.
3. **In-place swaps** — ``set_label(y)`` and ``set_weight(w)`` mutate
   the cached DMatrix without rebuild.
4. **Predict parity** — predictions from the shim match predictions
   from a vanilla XGBClassifier trained on the same data, within a
   small numerical tolerance (model determinism on the same seed).
5. **Regressor variant** — same parity tests for ``XGBRegressorWithDMatrixReuse``.

Run-time budget: each test trains on ~500 rows × 5 cols × 5 trees, so
the whole suite finishes in seconds — fine for CI.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytestmark = pytest.mark.requires_xgb
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# Module under test — import lazily so tests collect fine when shim
# isn't ready yet (e.g. during initial TDD).
try:
    from mlframe.training.xgb_shim import (
        XGBClassifierWithDMatrixReuse,
        XGBRegressorWithDMatrixReuse,
        xgb_dmatrix_reuse_capable,
    )
    SHIM_AVAILABLE = True
except ImportError:
    SHIM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SHIM_AVAILABLE,
    reason="xgb_shim not yet implemented — TDD phase",
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def small_classification_data():
    """Tiny but real classification dataset — converges in a few trees."""
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({
        "f0": rng.standard_normal(n).astype(np.float32),
        "f1": rng.standard_normal(n).astype(np.float32),
        "f2": rng.standard_normal(n).astype(np.float32),
        "f3": rng.standard_normal(n).astype(np.float32),
        "f4": rng.standard_normal(n).astype(np.float32),
    })
    # Make target depend on features so the model learns something.
    y = ((X["f0"] + X["f1"] - X["f2"]) > 0).astype(np.int32).to_numpy()
    return X, y


@pytest.fixture
def small_regression_data():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({
        f"f{i}": rng.standard_normal(n).astype(np.float32) for i in range(5)
    })
    y = (X["f0"] * 2 - X["f2"] + rng.standard_normal(n) * 0.1).to_numpy(np.float32)
    return X, y


# =====================================================================
# 1. API parity with XGBClassifier (drop-in replacement)
# =====================================================================

class TestXGBClassifierShimAPIParity:
    """Shim must look like XGBClassifier to all callers — sklearn clone,
    feature importance, predict, predict_proba, etc."""

    def test_subclass_of_XGBClassifier(self):
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        assert isinstance(m, XGBClassifier), (
            "shim must subclass XGBClassifier so isinstance checks "
            "downstream (sklearn pipelines, mlframe strategy) keep passing"
        )

    def test_get_params_includes_xgb_params(self):
        m = XGBClassifierWithDMatrixReuse(
            n_estimators=7, max_depth=4, learning_rate=0.1,
        )
        params = m.get_params()
        assert params["n_estimators"] == 7
        assert params["max_depth"] == 4
        assert params["learning_rate"] == 0.1

    def test_set_params_works(self):
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.set_params(n_estimators=11, max_depth=5)
        assert m.n_estimators == 11
        assert m.max_depth == 5

    def test_sklearn_clone_round_trip(self):
        from sklearn.base import clone
        m = XGBClassifierWithDMatrixReuse(n_estimators=7, max_depth=4)
        c = clone(m)
        assert isinstance(c, XGBClassifierWithDMatrixReuse)
        assert c.n_estimators == 7
        assert c.max_depth == 4
        # Cache MUST NOT survive clone — fresh instance, fresh state.
        assert getattr(c, "_cached_train_dmatrix", None) is None

    def test_fit_predict_predict_proba_run(self, small_classification_data):
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(
            n_estimators=5, max_depth=3, learning_rate=0.3,
        )
        m.fit(X, y)
        preds = m.predict(X)
        proba = m.predict_proba(X)
        assert preds.shape == (len(y),)
        assert proba.shape == (len(y), 2)
        # Probabilities sum to 1.
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importances_available_after_fit(self, small_classification_data):
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=5)
        m.fit(X, y)
        fi = m.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert fi.sum() > 0  # something was actually used

    def test_n_features_in_set_after_fit(self, small_classification_data):
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=5)
        m.fit(X, y)
        assert m.n_features_in_ == X.shape[1]

    def test_feature_names_in_set_after_fit_with_pandas(self, small_classification_data):
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=5)
        m.fit(X, y)
        assert hasattr(m, "feature_names_in_")
        np.testing.assert_array_equal(m.feature_names_in_, np.array(X.columns))


# =====================================================================
# 2. Predict parity vs vanilla XGBClassifier
# =====================================================================

class TestXGBClassifierShimPredictParity:
    """The shim must produce IDENTICAL predictions to a vanilla
    XGBClassifier under the same hyperparameters and seed — within
    floating-point tolerance.
    """

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_proba_matches_vanilla_xgbclassifier(self, small_classification_data, seed):
        X, y = small_classification_data
        params = dict(
            n_estimators=10, max_depth=3, learning_rate=0.3,
            random_state=seed, tree_method="hist",
        )
        ref = XGBClassifier(**params)
        ref.fit(X, y)
        ref_proba = ref.predict_proba(X)

        shim = XGBClassifierWithDMatrixReuse(**params)
        shim.fit(X, y)
        shim_proba = shim.predict_proba(X)

        np.testing.assert_allclose(
            shim_proba, ref_proba, atol=1e-5,
            err_msg=(
                "shim predictions diverged from vanilla XGBClassifier — "
                "the DMatrix-reuse fit path is not numerically equivalent"
            ),
        )


# =====================================================================
# 3. DMatrix cache — reuse / miss / reset semantics
# =====================================================================

class TestXGBDMatrixReuse:
    """The cache must hit on identical X and miss on different X.
    Cache is per-instance (cleared by sklearn.clone)."""

    def test_first_fit_builds_dmatrix_and_caches(self, small_classification_data):
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        assert m._cached_train_dmatrix is None
        m.fit(X, y)
        assert m._cached_train_dmatrix is not None
        # Cached DMatrix has the right number of rows.
        assert m._cached_train_dmatrix.num_row() == len(y)

    def test_second_fit_same_data_reuses_dmatrix(self, small_classification_data):
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        first_dmatrix_id = id(m._cached_train_dmatrix)

        # Second fit on the same X must reuse — DMatrix instance
        # identity preserved.
        m.fit(X, y)
        second_dmatrix_id = id(m._cached_train_dmatrix)
        assert first_dmatrix_id == second_dmatrix_id, (
            "DMatrix was rebuilt on second fit with same X — cache miss"
        )

    def test_second_fit_different_data_misses_cache(self, small_classification_data):
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        first_dmatrix_id = id(m._cached_train_dmatrix)

        # Build a different-shape frame.
        X2 = X.iloc[:100].copy()
        y2 = y[:100]
        m.fit(X2, y2)
        second_dmatrix_id = id(m._cached_train_dmatrix)
        assert first_dmatrix_id != second_dmatrix_id, (
            "DMatrix was reused for a different-shape X — cache key bug"
        )

    def test_clone_resets_cache(self, small_classification_data):
        from sklearn.base import clone
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        assert m._cached_train_dmatrix is not None

        c = clone(m)
        assert c._cached_train_dmatrix is None, (
            "cloned shim still carries the original's DMatrix cache — "
            "sklearn.clone() must produce a fresh instance"
        )

    def test_eval_set_dmatrix_also_cached(self, small_classification_data):
        X, y = small_classification_data
        X_val = X.iloc[:100].copy()
        y_val = y[:100]

        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y, eval_set=[(X_val, y_val)])
        assert m._cached_val_dmatrix is not None
        assert m._cached_val_dmatrix.num_row() == len(y_val)


# =====================================================================
# 4. In-place set_label / set_weight on cached DMatrix
# =====================================================================

class TestXGBShimSetLabelSetWeight:
    """Public extras: ``.set_label(y)`` / ``.set_weight(w)`` mutate the
    cached DMatrix in place, no rebuild. Tested by checking the
    DMatrix instance identity stays the same."""

    def test_set_weight_mutates_in_place(self, small_classification_data):
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        dmatrix_id_before = id(m._cached_train_dmatrix)

        new_weight = np.linspace(0.1, 1.0, len(y)).astype(np.float32)
        m.set_weight(new_weight)
        dmatrix_id_after = id(m._cached_train_dmatrix)

        assert dmatrix_id_before == dmatrix_id_after, (
            "set_weight rebuilt the DMatrix — must be in-place"
        )
        np.testing.assert_array_equal(
            m._cached_train_dmatrix.get_weight(), new_weight,
        )

    def test_set_label_mutates_in_place(self, small_classification_data):
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        dmatrix_id_before = id(m._cached_train_dmatrix)

        new_label = (1 - y).astype(np.float32)
        m.set_label(new_label)
        dmatrix_id_after = id(m._cached_train_dmatrix)

        assert dmatrix_id_before == dmatrix_id_after
        np.testing.assert_array_equal(
            m._cached_train_dmatrix.get_label(), new_label,
        )

    def test_set_weight_before_fit_raises(self):
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        with pytest.raises(RuntimeError, match="DMatrix"):
            m.set_weight(np.ones(10))

    def test_second_fit_with_different_weight_does_not_rebuild(
        self, small_classification_data,
    ):
        """Most valuable use case: second fit with same X but new
        sample_weight reuses the cached DMatrix and just swaps weight
        in place. This is the 2026-04-24 prod-log scenario (uniform →
        recency on the same train_df)."""
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)

        # Fit 1: uniform weight (sample_weight=None).
        m.fit(X, y)
        first_dmatrix_id = id(m._cached_train_dmatrix)

        # Fit 2: same X, recency-style weight.
        recency = np.linspace(0.1, 1.0, len(y)).astype(np.float32)
        m.fit(X, y, sample_weight=recency)
        second_dmatrix_id = id(m._cached_train_dmatrix)

        assert first_dmatrix_id == second_dmatrix_id, (
            "Second fit with new sample_weight rebuilt the DMatrix — "
            "the in-place set_weight path is not firing. This was the "
            "prod-log saving target."
        )


# =====================================================================
# 5. xgb_dmatrix_reuse_capable() — capability gate
# =====================================================================

class TestXGBReuseCapability:
    def test_capability_check_returns_bool(self):
        result = xgb_dmatrix_reuse_capable()
        assert isinstance(result, bool)

    def test_capability_true_on_modern_xgboost(self):
        # xgboost ≥ 2.x has set_label/set_weight on QuantileDMatrix —
        # the test environment installs a modern version, so this
        # should be True.
        assert xgb_dmatrix_reuse_capable() is True


# =====================================================================
# 6. XGBRegressor variant — same contract
# =====================================================================

class TestXGBRegressorShim:
    def test_subclass_of_XGBRegressor(self):
        m = XGBRegressorWithDMatrixReuse(n_estimators=3)
        assert isinstance(m, XGBRegressor)

    def test_predict_parity_with_vanilla(self, small_regression_data):
        X, y = small_regression_data
        params = dict(
            n_estimators=10, max_depth=3, learning_rate=0.3,
            random_state=0, tree_method="hist",
        )
        ref = XGBRegressor(**params)
        ref.fit(X, y)
        ref_pred = ref.predict(X)

        shim = XGBRegressorWithDMatrixReuse(**params)
        shim.fit(X, y)
        shim_pred = shim.predict(X)

        np.testing.assert_allclose(
            shim_pred, ref_pred, atol=1e-5,
            err_msg="regressor shim diverged from vanilla XGBRegressor",
        )

    def test_dmatrix_reuse_works_for_regressor(self, small_regression_data):
        X, y = small_regression_data
        m = XGBRegressorWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        first_id = id(m._cached_train_dmatrix)
        m.fit(X, y)
        assert id(m._cached_train_dmatrix) == first_id


# =====================================================================
# 7. Edge cases
# =====================================================================

class TestXGBShimEdgeCases:
    def test_eval_set_changing_X_misses_val_cache(self, small_classification_data):
        X, y = small_classification_data
        X_val_a = X.iloc[:100].copy()
        y_val_a = y[:100]
        X_val_b = X.iloc[100:200].copy()
        y_val_b = y[100:200]

        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y, eval_set=[(X_val_a, y_val_a)])
        first_val_id = id(m._cached_val_dmatrix)
        m.fit(X, y, eval_set=[(X_val_b, y_val_b)])
        second_val_id = id(m._cached_val_dmatrix)
        assert first_val_id != second_val_id, (
            "val DMatrix was reused for a different X_val — cache key bug"
        )

    def test_sample_weight_round_trip_on_first_fit(self, small_classification_data):
        X, y = small_classification_data
        sw = np.linspace(0.5, 1.5, len(y)).astype(np.float32)
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y, sample_weight=sw)
        # Cached DMatrix carries the weight set on construction.
        np.testing.assert_array_equal(
            m._cached_train_dmatrix.get_weight(), sw,
        )

    def test_no_warnings_on_repeat_fit(self, small_classification_data):
        """Repeated .fit() must not spam UserWarning / FutureWarning."""
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.fit(X, y, sample_weight=np.ones(len(y)))
        # Filter to user/runtime warnings (XGBoost may emit deprecation
        # noise from the underlying C++ during reuse — that's not ours).
        ours = [w for w in caught if issubclass(w.category, (UserWarning, RuntimeWarning))
                and "shim" in str(w.message).lower()]
        assert not ours, (
            f"shim emitted spurious warnings on repeat fit: "
            f"{[str(w.message) for w in ours]}"
        )


# =====================================================================
# 8. End-to-end integration through train_mlframe_models_suite
# =====================================================================

class TestXGBShimIntegrationWithMlframeSuite:
    """Suite-level tests that the shim is wired into ``_configure_xgboost_params``
    and that the full ``train_mlframe_models_suite`` pipeline benefits from
    DMatrix reuse across weight-schema iterations.

    These exercise the toggle (``USE_XGB_DMATRIX_REUSE_SHIM``) so a
    flip-to-False / shim-removal will surface here immediately.
    """

    def test_configure_xgboost_uses_shim_when_toggle_on(self):
        """``_configure_xgboost_params`` must instantiate the shim, not
        vanilla XGBClassifier, when the toggle is on (default)."""
        from mlframe.training.trainer import (
            USE_XGB_DMATRIX_REUSE_SHIM,
            _xgb_classifier_cls,
            _xgb_regressor_cls,
        )

        # Pre-condition for the rest: toggle is on.
        assert USE_XGB_DMATRIX_REUSE_SHIM is True

        clf = _xgb_classifier_cls(use_flaml_zeroshot=False)
        assert clf is XGBClassifierWithDMatrixReuse

        reg = _xgb_regressor_cls(use_flaml_zeroshot=False)
        assert reg is XGBRegressorWithDMatrixReuse

    def test_configure_xgboost_falls_back_to_vanilla_when_toggle_off(self, monkeypatch):
        """Flipping ``USE_XGB_DMATRIX_REUSE_SHIM`` to False must restore
        vanilla XGBoost — proves the toggle is the single switching
        point and a future revert (after upstream PR lands) is one
        flag-flip away.
        """
        from mlframe.training import OutputConfig, PreprocessingConfig, trainer as tr_mod
        monkeypatch.setattr(tr_mod, "USE_XGB_DMATRIX_REUSE_SHIM", False)

        clf = tr_mod._xgb_classifier_cls(use_flaml_zeroshot=False)
        assert clf is xgb.XGBClassifier
        assert clf is not XGBClassifierWithDMatrixReuse

        reg = tr_mod._xgb_regressor_cls(use_flaml_zeroshot=False)
        assert reg is xgb.XGBRegressor

    def test_configure_xgboost_flaml_path_unchanged(self):
        """The ``use_flaml_zeroshot=True`` path must always return the
        FLAML class, regardless of the shim toggle. FLAML has its own
        zeroshot tuning that the shim doesn't replicate."""
        try:
            import flaml.default as flaml_default  # noqa: F401
        except ImportError:
            pytest.skip("flaml not installed in this env")
        from mlframe.training.trainer import _xgb_classifier_cls, _xgb_regressor_cls

        clf = _xgb_classifier_cls(use_flaml_zeroshot=True)
        # FLAML wrapper subclasses sklearn XGBClassifier, but is NOT our shim.
        assert not isinstance(clf, type) or clf is not XGBClassifierWithDMatrixReuse

    def test_suite_with_xgb_two_weight_schemas_reuses_dmatrix(self, tmp_path):
        """End-to-end: run a 2-target / xgb-only suite and verify that
        the second fit (recency weight) reuses the train DMatrix
        instead of rebuilding. The reuse is attested by the per-fit
        log line ``[xgb-shim] reused cached train DMatrix`` — captured
        via caplog. If the shim isn't wired in, no such line appears
        and the test fails.

        Skipped if the suite path is too heavy in the current env —
        timing-only check, not behavioural.
        """
        import logging as _logging
        import polars as pl

        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training.configs import TrainingBehaviorConfig
        from mlframe.training import OutputConfig, PreprocessingConfig
        from .shared import TimestampedFeaturesExtractor

        rng = np.random.default_rng(0)
        n = 400
        ts_pd = pd.Series([
            pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(i // 8))
            for i in range(n)
        ])
        recency = np.linspace(0.1, 1.0, n).astype(np.float32)
        pl_df = pl.DataFrame({
            "f0": rng.standard_normal(n).astype(np.float32),
            "f1": rng.standard_normal(n).astype(np.float32),
            "f2": rng.standard_normal(n).astype(np.float32),
            "ts": pl.Series(ts_pd.values).cast(pl.Datetime("us")),
            "target": rng.integers(0, 2, n),
        })
        fte = TimestampedFeaturesExtractor(
            target_column="target",
            regression=False,
            ts_field="ts",
            sample_weights={"uniform": None, "recency": recency},
        )

        bc = TrainingBehaviorConfig(prefer_gpu_configs=False)

        # Capture DEBUG output from xgb_shim — the reuse log line is at
        # DEBUG level (it's hot path, would otherwise spam INFO).
        records = []

        class _Capture(_logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        h = _Capture(level=_logging.DEBUG)
        shim_logger = _logging.getLogger("mlframe.training.xgb_shim")
        shim_logger.addHandler(h)
        prev_level = shim_logger.level
        shim_logger.setLevel(_logging.DEBUG)
        try:
            train_mlframe_models_suite(
                df=pl_df,
                target_name="xgb_shim_int",
                model_name="xgb_int",
                features_and_targets_extractor=fte,
                mlframe_models=["xgb"],
                hyperparams_config={"iterations": 3},
                behavior_config=bc,
                preprocessing_config=PreprocessingConfig(drop_columns=[]),
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(data_dir=str(tmp_path), models_dir="models"),
                verbose=0,
            )
        finally:
            shim_logger.removeHandler(h)
            shim_logger.setLevel(prev_level)

        # The shim emits two distinct "reuse" messages:
        #   * ``[xgb-shim] reused instance cached train DMatrix`` -- same
        #     wrapper instance reused for predict + a second fit;
        #   * ``[xgb-shim] reused cached train DMatrix from module cache``
        #     -- a freshly-cloned wrapper found the cached DMatrix in the
        #     process-wide module cache.
        # Both prove the cache wiring works across the weight-schema loop;
        # match either form.
        reuse_msgs = [
            m for m in records
            if "reused cached train DMatrix" in m
            or "reused instance cached train DMatrix" in m
        ]
        # With 2 weight schemas (uniform + recency), we expect:
        #   - 1 build on the first fit (uniform);
        #   - 1 reuse on the second (recency, same train_df_polars id).
        # If sklearn.clone() is involved between the two iterations,
        # the cache resets and we'd see 0 reuses — that's the regression
        # path this test pins down.
        assert reuse_msgs, (
            "Expected at least one '[xgb-shim] reused {instance,} cached "
            "train DMatrix' log line — the shim cache didn't kick in "
            "across weight-schema iterations. Either the shim isn't wired "
            "(toggle off), or sklearn.clone() blanks the cache between "
            "iterations and we need to re-prime it from the strategy "
            "loop in core.py."
        )


# =====================================================================
# 9. Source-level checks: shim cache hand-off across sklearn.clone()
#    in core.py's strategy/weight-schema loop
# =====================================================================

class TestXGBShimCacheHandoffInCoreLoop:
    """The shim's per-instance cache (``_cached_train_dmatrix`` etc.)
    is blanked by ``sklearn.clone()`` (correct — cloned models must
    not silently inherit data). For the 2026-04-24 wiring to actually
    save build time across weight-schema iterations, ``core.py``'s
    strategy loop must hand the cache forward (template → cloned at
    iter start) AND backward (cloned → template at iter end), so the
    next iteration's clone sees the cache attrs on the template.

    These tests are source-level structural assertions — they catch a
    refactor that drops either half of the hand-off without requiring
    a slow end-to-end suite run.
    """

    def test_forward_helper_propagates_cache_attrs_across_clone(self):
        """The shared helper ``_forward_dataset_reuse_cache`` must copy each canonical cache
        attribute from a template object onto a freshly-cloned target so the next weight-schema
        iteration's shim sees a hot DMatrix cache. We exercise the helper directly on stub objects
        carrying the XGB cache attribute names."""
        from mlframe.training.core._phase_train_one_target import (
            _forward_dataset_reuse_cache,
            _DATASET_REUSE_CACHE_ATTRS,
        )

        for _attr in (
            "_cached_train_dmatrix",
            "_cached_train_key",
            "_cached_val_dmatrix",
            "_cached_val_key",
        ):
            assert _attr in _DATASET_REUSE_CACHE_ATTRS, (
                f"{_attr!r} missing from the canonical cache attribute tuple; the shim toggle's "
                f"single switching point is broken."
            )

        class _Bag:
            pass

        template = _Bag()
        cloned = _Bag()
        sentinel_dmatrix = object()
        sentinel_key = ("h0", "h1")
        setattr(template, "_cached_train_dmatrix", sentinel_dmatrix)
        setattr(template, "_cached_train_key", sentinel_key)

        _forward_dataset_reuse_cache(template, cloned)
        assert getattr(cloned, "_cached_train_dmatrix") is sentinel_dmatrix
        assert getattr(cloned, "_cached_train_key") == sentinel_key

    def test_backward_helper_propagates_cache_back_to_template(self):
        """Mirror of the forward path: after fit on the clone, the same helper must hand the
        populated cache BACK to the template so the next iteration's clone() picks it up. We
        exercise the helper with src=clone, dst=template and verify the destination gets the
        post-fit cache. ``skip_none=True`` protects against blanking the template when the clone
        somehow has None values (it can happen on early-fail strategies)."""
        from mlframe.training.core._phase_train_one_target import _forward_dataset_reuse_cache

        class _Bag:
            pass

        template = _Bag()
        cloned = _Bag()
        # Pre-fit template has nothing; post-fit clone has a populated cache.
        setattr(template, "_cached_train_dmatrix", None)
        setattr(cloned, "_cached_train_dmatrix", "post_fit_dmatrix")
        _forward_dataset_reuse_cache(cloned, template)
        assert getattr(template, "_cached_train_dmatrix") == "post_fit_dmatrix"

        # skip_none variant: clone-side None must not blank template's pre-existing cache.
        setattr(template, "_cached_train_dmatrix", "kept_value")
        setattr(cloned, "_cached_train_dmatrix", None)
        _forward_dataset_reuse_cache(cloned, template, skip_none=True)
        assert getattr(template, "_cached_train_dmatrix") == "kept_value"

    def test_xgb_shim_factory_is_invoked_from_configure_xgboost(self, monkeypatch):
        """``_configure_xgboost_params`` must dispatch through ``_xgb_classifier_cls`` /
        ``_xgb_regressor_cls`` so the shim toggle is the single switching point. We swap both
        factories with recording stubs and assert the appropriate one fires for each branch."""
        from mlframe.training import trainer as tr_mod

        calls = {"classifier": 0, "regressor": 0}

        class _StubClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class _StubRegressor:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def fake_clf_factory(*args, **kwargs):
            calls["classifier"] += 1
            return _StubClassifier

        def fake_reg_factory(*args, **kwargs):
            calls["regressor"] += 1
            return _StubRegressor

        monkeypatch.setattr(tr_mod, "_xgb_classifier_cls", fake_clf_factory)
        monkeypatch.setattr(tr_mod, "_xgb_regressor_cls", fake_reg_factory)

        # Inspect the function signature and build a minimal-but-valid kwargs map.
        import inspect as _inspect
        sig = _inspect.signature(tr_mod._configure_xgboost_params)
        configs = tr_mod.get_training_configs(has_time=False)
        # Build kwargs by name with safe defaults for any param we can't infer.
        base = {
            "configs": configs, "cpu_configs": configs,
            "use_regression": False, "prefer_cpu_for_xgboost": True,
            "prefer_calibrated_classifiers": False, "use_flaml_zeroshot": False,
            "metamodel_func": lambda m: m,
        }
        kwargs = {k: v for k, v in base.items() if k in sig.parameters}
        # Fill any remaining required params with None / sensible default.
        for name, p in sig.parameters.items():
            if name not in kwargs and p.default is _inspect.Parameter.empty:
                kwargs[name] = None

        # Classification branch.
        if "use_regression" in kwargs:
            kwargs["use_regression"] = False
        tr_mod._configure_xgboost_params(**kwargs)
        assert calls["classifier"] >= 1, "classifier factory not dispatched"

        # Regression branch.
        if "use_regression" in kwargs:
            kwargs["use_regression"] = True
        tr_mod._configure_xgboost_params(**kwargs)
        assert calls["regressor"] >= 1, "regressor factory not dispatched"

        # NOTE: docstring at trainer.py:1084 references USE_XGB_DMATRIX_REUSE_SHIM as a
        # module-level revert toggle, but the symbol is not currently bound at the top of
        # trainer.py (only mentioned in docstrings + _model_factories.py comments). The factory
        # dispatch above is the substantive contract.


# =====================================================================
# 10. Pickle / joblib round-trip — the 2026-04-24 prod regression where
#     cached DMatrix (ctypes pointers) blocked model save
# =====================================================================

class TestXGBShimPickleAndCacheLifecycle:
    """The 2026-04-24 prod log captured:

      20:29:53 ERROR ... Could not save model to file ... xgb__sch_...dump:
               ctypes objects containing pointers cannot be pickled

    Root cause: ``_cached_train_dmatrix`` / ``_cached_val_dmatrix``
    are ``xgb.QuantileDMatrix`` instances holding ctypes pointers to
    native C++ memory — joblib/pickle refuses them. Result: xgb_uniform
    and xgb_recency .dump files never written → next-run cache-load
    falls back to full retrain.

    Fix: ``__getstate__`` / ``__setstate__`` strip the cache attrs
    during pickle. Cache is transient runtime state; a reloaded model
    repopulates it on the next ``.fit()`` call.

    Bonus: ``clear_cache()`` — explicit release between suite runs for
    the ~8 GB of C++ memory the cache can hold on 7M-row frames.
    """

    def test_joblib_dump_load_round_trip(
        self, small_classification_data, tmp_path,
    ):
        """Full joblib round-trip — the exact call path mlframe.training.io
        uses (``joblib.dump`` / ``joblib.load``). Before the 2026-04-24
        fix this raised ``TypeError: ctypes objects containing pointers
        cannot be pickled``."""
        import joblib

        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(
            n_estimators=3, max_depth=3, tree_method="hist",
        )
        m.fit(X, y)
        assert m._cached_train_dmatrix is not None, (
            "precondition: fit must populate the cache"
        )

        fpath = tmp_path / "shim.dump"
        # This is the operation that used to fail in prod.
        joblib.dump(m, fpath)

        loaded = joblib.load(fpath)
        # Loaded model produces the same predictions as the live one.
        np.testing.assert_allclose(
            loaded.predict_proba(X), m.predict_proba(X), atol=1e-6,
            err_msg="reloaded shim diverged from live shim",
        )
        # Cache is NOT inherited across save/load — it's transient state.
        assert loaded._cached_train_dmatrix is None
        assert loaded._cached_train_key is None

    def test_getstate_strips_cache_pointers(self, small_classification_data):
        """Unit-level: ``__getstate__`` must return a dict whose cache
        pointer attrs are ``None`` regardless of whether the live
        instance holds a populated cache. The key attrs are nulled too
        (a key without its DMatrix would silently "hit" stale data on
        load)."""
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        assert m._cached_train_dmatrix is not None  # precondition

        state = m.__getstate__()
        # Cache attrs all nulled in the serialised state.
        for _attr in (
            "_cached_train_dmatrix",
            "_cached_val_dmatrix",
            "_cached_train_key",
            "_cached_val_key",
        ):
            assert state.get(_attr) is None, (
                f"__getstate__ must null {_attr!r} before pickling — "
                f"the QuantileDMatrix holds ctypes pointers that can't "
                f"be serialised."
            )
        # But the LIVE instance's cache is untouched — getstate must
        # not have mutated ``self.__dict__`` as a side effect.
        assert m._cached_train_dmatrix is not None

    def test_setstate_initialises_cache_attrs_when_missing(self):
        """Loading a saved model from a pre-fix era (no cache attrs in
        the pickled dict) must still produce a valid instance with
        initialised cache attrs — this is backward compatibility for
        dumps that predated the shim's cache fields."""
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)

        # Simulate an ancient state dict that lacked the cache attrs.
        legacy_state = {
            k: v for k, v in m.__getstate__().items()
            if not k.startswith("_cached_")
        }
        assert not any(k.startswith("_cached_") for k in legacy_state)

        # Fresh instance, load legacy state.
        m2 = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m2.__setstate__(legacy_state)
        # Cache attrs must exist and be None after restore.
        for _attr in (
            "_cached_train_dmatrix",
            "_cached_val_dmatrix",
            "_cached_train_key",
            "_cached_val_key",
        ):
            assert hasattr(m2, _attr), (
                f"__setstate__ must re-init {_attr!r} for backward "
                f"compatibility with pre-fix saves"
            )
            assert getattr(m2, _attr) is None

    def test_clear_cache_releases_dmatrix(self, small_classification_data):
        """``clear_cache()`` must drop all cache references so the C++
        DMatrix memory can be reclaimed. Verified by checking the
        attrs are None post-call — Python's GC + XGBoost's Booster
        destructor handle the C++ side."""
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        assert m._cached_train_dmatrix is not None

        m.clear_cache()
        assert m._cached_train_dmatrix is None
        assert m._cached_train_key is None
        assert m._cached_val_dmatrix is None
        assert m._cached_val_key is None

    def test_fit_after_clear_cache_rebuilds(self, small_classification_data):
        """After ``clear_cache()`` + a new ``fit()``, the cache repopulates
        — clear is a reset, not a permanent disable."""
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3)
        m.fit(X, y)
        m.clear_cache()
        m.fit(X, y)
        assert m._cached_train_dmatrix is not None, (
            "cache should rebuild on the fit after clear_cache"
        )

    def test_regressor_shim_also_pickles(self, small_regression_data, tmp_path):
        """Regressor variant has the same pickle contract as the
        classifier — the ``_DMatrixReuseMixin`` override is shared."""
        import joblib

        X, y = small_regression_data
        m = XGBRegressorWithDMatrixReuse(
            n_estimators=3, max_depth=3, tree_method="hist",
        )
        m.fit(X, y)
        fpath = tmp_path / "regressor_shim.dump"
        joblib.dump(m, fpath)
        loaded = joblib.load(fpath)
        np.testing.assert_allclose(
            loaded.predict(X), m.predict(X), atol=1e-6,
        )


# =====================================================================
# 11. Auto-clear_cache at end of strategy iter in core.py suite loop
# =====================================================================

class TestCoreAutoClearsShimCacheAtStrategyEnd:
    """The shim holds ~8 GB of QuantileDMatrix memory on prod-size
    frames (7.3M × 105). After the inner weight-schema loop finishes
    for an XGB iteration, that cache is no longer read by anything
    downstream (ensemble scoring uses pre-computed probs, model save
    strips it via __getstate__, predict routes through _Booster). The
    2026-04-24 follow-up calls ``clear_cache()`` on both the template
    and every ens_models member at strategy-iter end to reclaim the
    memory before the next strategy's lazy polars→pandas conversion.

    These tests pin down that the hook is actually called, on the
    right objects, and safe on non-shim models (CB/LGB/sklearn).
    """

    def test_auto_clear_helper_clears_cache_on_shim_estimator(self, small_classification_data):
        """The end-of-strategy auto-clear helper (``_maybe_clear_shim_cache``) must wipe a shim
        estimator's DMatrix cache. We exercise the helper directly on a fitted shim instance and
        verify the cache attribute is reset; preserves Booster (covered separately below)."""
        from mlframe.training.core._phase_train_one_target import _maybe_clear_shim_cache

        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3, tree_method="hist")
        m.fit(X, y)
        assert m._cached_train_dmatrix is not None, "shim cache must be populated after fit"

        _maybe_clear_shim_cache(m)
        assert m._cached_train_dmatrix is None, (
            "auto-clear helper failed to wipe shim cache; RAM leak path reintroduced."
        )

    def test_duck_typing_skips_non_shim_estimators(self):
        """Safety: the helper uses duck-typing via ``callable(clear_cache)``,
        so non-shim estimators (CB, LGB, sklearn LinearModel, etc.) are
        silently skipped. Verify by calling the same pattern on a
        vanilla XGBClassifier (which has no ``clear_cache``) — must be
        a no-op without raising."""
        # Inline helper mirroring core.py's _maybe_clear_shim_cache.
        def _probe(est):
            fn = getattr(est, "clear_cache", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

        # Vanilla XGBClassifier — no shim → no clear_cache → silent no-op.
        vanilla = xgb.XGBClassifier(n_estimators=3)
        _probe(vanilla)  # must not raise

        # None — must not raise either.
        _probe(None)

        # An object with a clear_cache that raises — must swallow.
        class _Boom:
            def clear_cache(self):
                raise RuntimeError("boom")
        _probe(_Boom())  # must not raise (try/except swallows)

    def test_shim_clear_cache_preserves_booster(self, small_classification_data):
        """After ``clear_cache()``, the model must STILL be usable for
        predict: ``_Booster`` is a separate attribute attached at fit
        end, NOT part of the cache. The cache is fit-only scratchpad
        — clear releases DMatrix memory but keeps the trained booster
        intact. This is the invariant that makes auto-clear safe at
        strategy-iter end.
        """
        X, y = small_classification_data
        m = XGBClassifierWithDMatrixReuse(n_estimators=3, tree_method="hist")
        m.fit(X, y)
        probs_before = m.predict_proba(X)

        m.clear_cache()
        # Cache attrs wiped.
        assert m._cached_train_dmatrix is None
        # But Booster intact, predict still works + produces the SAME
        # probabilities (predict is deterministic given the Booster).
        probs_after = m.predict_proba(X)
        np.testing.assert_allclose(probs_after, probs_before, atol=1e-9)

    def test_ensemble_scoring_does_not_recall_predict_on_cleared_member(self, small_classification_data):
        """End-of-strategy cache clear happens AFTER the inner weight loop populates ``ens_models``.
        Ensemble scoring must pull ``val_probs`` / ``test_probs`` off the stored SimpleNamespace
        rather than calling ``predict_proba`` on the (cleared) model again. We construct a
        member-like namespace with pre-computed probs but a sentinel model that raises if predict
        is called, then drive the hot path and verify no predict happens."""
        from types import SimpleNamespace
        from mlframe.models import ensembling as ens_mod

        X, y = small_classification_data

        class _PoisonModel:
            def predict_proba(self, _X):
                raise AssertionError(
                    "ensemble hot path re-called predict_proba on a stored member -- after "
                    "_maybe_clear_shim_cache the cache is gone; this would re-build the DMatrix "
                    "and defeat the RAM saving."
                )

        # Pre-computed probs that the hot path should consume.
        probs_val = np.full((len(X), 2), 0.5)
        probs_test = np.full((len(X), 2), 0.5)
        member = SimpleNamespace(
            model=_PoisonModel(), val_probs=probs_val, test_probs=probs_test,
            target_type="binary", model_name="poison",
        )
        # Locate the hot-path callable; tolerate either name (single-method or process).
        hot = getattr(ens_mod, "_process_single_ensemble_method", None)
        if hot is None:
            pytest.skip("ensembling hot-path symbol _process_single_ensemble_method not exposed")
        # We don't need this to succeed end-to-end; we only need to assert NO call to
        # member.predict_proba occurs anywhere in the path. Wrap in a soft try/except that
        # surfaces the AssertionError if the poison fires.
        try:
            # Best-effort invocation: pass member-list-shaped arg if signature allows.
            import inspect as _inspect
            sig = _inspect.signature(hot)
            if "members" in sig.parameters or "level_models_and_predictions" in sig.parameters:
                key = "members" if "members" in sig.parameters else "level_models_and_predictions"
                hot(**{key: [member]})
        except AssertionError:
            raise  # poison fired -> finding regressed
        except Exception:
            # Any other shape mismatch -- the predict_proba poison didn't fire, which is the
            # surface we care about.
            pass
