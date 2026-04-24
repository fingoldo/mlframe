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
        from mlframe.training import trainer as tr_mod
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
                init_common_params={"drop_columns": [], "verbose": 0},
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                data_dir=str(tmp_path),
                models_dir="models",
                verbose=0,
            )
        finally:
            shim_logger.removeHandler(h)
            shim_logger.setLevel(prev_level)

        reuse_msgs = [m for m in records if "reused cached train DMatrix" in m]
        # With 2 weight schemas (uniform + recency), we expect:
        #   - 1 build on the first fit (uniform);
        #   - 1 reuse on the second (recency, same train_df_polars id).
        # If sklearn.clone() is involved between the two iterations,
        # the cache resets and we'd see 0 reuses — that's the regression
        # path this test pins down.
        assert reuse_msgs, (
            "Expected at least one '[xgb-shim] reused cached train DMatrix' "
            "log line — the shim cache didn't kick in across weight-schema "
            "iterations. Either the shim isn't wired (toggle off), or "
            "sklearn.clone() blanks the cache between iterations and we "
            "need to re-prime it from the strategy loop in core.py."
        )
