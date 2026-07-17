"""
Tests for training/strategies.py module.

Covers:
- ModelPipelineStrategy abstract base class
- TreeModelStrategy
- HGBStrategy
- NeuralNetStrategy
- LinearModelStrategy
- PipelineCache
- Strategy registry and get_strategy function
"""

import pytest
import warnings
from unittest.mock import MagicMock

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from mlframe.training.strategies import (
    ModelPipelineStrategy,
    TreeModelStrategy,
    HGBStrategy,
    NeuralNetStrategy,
    LinearModelStrategy,
    MODEL_STRATEGIES,
    get_strategy,
    PipelineCache,
)


# =============================================================================
# Strategy Property Tests
# =============================================================================


class TestTreeModelStrategy:
    """Tests for TreeModelStrategy."""

    def test_cache_key(self):
        """Test tree strategy cache key."""
        strategy = TreeModelStrategy()
        assert strategy.cache_key == "tree"

    def test_requires_scaling_false(self):
        """Test tree models don't require scaling."""
        strategy = TreeModelStrategy()
        assert strategy.requires_scaling is False

    def test_requires_encoding_false(self):
        """Test tree models don't require encoding."""
        strategy = TreeModelStrategy()
        assert strategy.requires_encoding is False

    def test_requires_imputation_false(self):
        """Test tree models don't require imputation."""
        strategy = TreeModelStrategy()
        assert strategy.requires_imputation is False

    def test_build_pipeline_returns_base_only(self):
        """Test that tree strategy returns only base pipeline."""
        strategy = TreeModelStrategy()
        base_pipeline = Pipeline([("step1", StandardScaler())])

        result = strategy.build_pipeline(
            base_pipeline=base_pipeline,
            cat_features=["cat1"],
            category_encoder=OrdinalEncoder(),
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
        )

        # Should return base pipeline unchanged
        assert result is base_pipeline

    def test_build_pipeline_none_base(self):
        """Test tree strategy with no base pipeline."""
        strategy = TreeModelStrategy()

        result = strategy.build_pipeline(
            base_pipeline=None,
            cat_features=["cat1"],
            category_encoder=OrdinalEncoder(),
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
        )

        assert result is None


class TestHGBStrategy:
    """Tests for HGBStrategy."""

    def test_cache_key(self):
        """Test HGB strategy cache key."""
        strategy = HGBStrategy()
        assert strategy.cache_key == "hgb"

    def test_requires_scaling_false(self):
        """Test HGB models don't require scaling."""
        strategy = HGBStrategy()
        assert strategy.requires_scaling is False

    def test_requires_encoding_true(self):
        """Test HGB models require encoding."""
        strategy = HGBStrategy()
        assert strategy.requires_encoding is True

    def test_requires_imputation_false(self):
        """Test HGB models don't require imputation."""
        strategy = HGBStrategy()
        assert strategy.requires_imputation is False

    def test_build_pipeline_with_encoding(self):
        """Test that HGB strategy adds encoding step."""
        strategy = HGBStrategy()
        encoder = OrdinalEncoder()

        result = strategy.build_pipeline(
            base_pipeline=None,
            cat_features=["cat1"],
            category_encoder=encoder,
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
        )

        assert result is not None
        assert isinstance(result, Pipeline)
        # Should have category encoder but not scaler or imputer
        step_names = [name for name, _ in result.steps]
        assert "ce" in step_names
        assert "scaler" not in step_names
        assert "imp" not in step_names

    def test_build_pipeline_no_cat_features(self):
        """Test HGB strategy with no categorical features."""
        strategy = HGBStrategy()

        result = strategy.build_pipeline(
            base_pipeline=None,
            cat_features=[],  # No categorical features
            category_encoder=OrdinalEncoder(),
            imputer=SimpleImputer(),
            scaler=StandardScaler(),
        )

        # Should return None or base_pipeline when no categorical features
        assert result is None


class TestNeuralNetStrategy:
    """Tests for NeuralNetStrategy."""

    def test_cache_key(self):
        """Test neural strategy cache key."""
        strategy = NeuralNetStrategy()
        assert strategy.cache_key == "neural"

    def test_requires_scaling_true(self):
        """Test neural networks require scaling."""
        strategy = NeuralNetStrategy()
        assert strategy.requires_scaling is True

    def test_requires_encoding_false_when_learnable_cat_embeddings(self):
        """With learnable cat embeddings ON (the default), the neural strategy must NOT target-encode cats -- they reach the MLP raw so its
        fit-boundary factorizer + nn.Embedding can index them -- so requires_encoding is False."""
        strategy = NeuralNetStrategy()
        assert strategy.use_learnable_cat_embeddings is True
        assert strategy.requires_encoding is False

    def test_requires_encoding_true_when_learnable_cat_embeddings_off(self):
        """Flipping the knob OFF restores the legacy CatBoostEncoder path -> requires_encoding True."""
        strategy = NeuralNetStrategy()
        try:
            type(strategy).use_learnable_cat_embeddings = False
            assert strategy.requires_encoding is True
        finally:
            type(strategy).use_learnable_cat_embeddings = True

    def test_requires_imputation_true(self):
        """Test neural networks require imputation."""
        strategy = NeuralNetStrategy()
        assert strategy.requires_imputation is True

    def test_build_pipeline_skips_encoder_keeps_numeric_imp_scale(self):
        """Learnable cat embeddings ON: the CatBoostEncoder step ('ce') is SKIPPED (cats reach the MLP raw), but imputation + scaling steps
        stay (restricted to the numeric block via _NumericOnlyTransformer)."""
        strategy = NeuralNetStrategy()
        encoder = OrdinalEncoder()
        imputer = SimpleImputer()
        scaler = StandardScaler()

        result = strategy.build_pipeline(
            base_pipeline=None,
            cat_features=["cat1"],
            category_encoder=encoder,
            imputer=imputer,
            scaler=scaler,
        )

        assert result is not None
        assert isinstance(result, Pipeline)
        step_names = [name for name, _ in result.steps]
        assert "ce" not in step_names  # cats reach the MLP raw -> no target encoder
        assert "imp" in step_names
        assert "scaler" in step_names

    def test_numeric_only_transformer_passes_string_cats_around_scaler(self):
        """Regression (bug-hunt): with cat_features NOT threaded (empty), _NumericOnlyTransformer must still route ONLY numeric columns to the
        inner scaler/imputer -- a raw string column passes through untouched. Pre-fix it sent the string column to StandardScaler ->
        'could not convert string to float' (the 2nd-largest fuzz failure class at 1k rows)."""
        import pandas as pd
        import numpy as np
        from mlframe.training.strategies.base import _NumericOnlyTransformer

        X = pd.DataFrame({"num": np.arange(10.0), "cat_raw": list("ABABABABAB")})  # string col, NO named cats
        t = _NumericOnlyTransformer(StandardScaler(), cat_features=[])
        Xt = t.fit_transform(X)  # pre-fix: ValueError could not convert string to float: 'A'
        assert "cat_raw" in Xt.columns  # string col passed through untouched
        assert abs(float(np.asarray(Xt["num"]).mean())) < 1e-6  # numeric col was scaled

    def test_build_pipeline_full_when_learnable_cat_embeddings_off(self):
        """Knob OFF: the legacy full preprocessing (encoder + imputer + scaler) is restored verbatim."""
        strategy = NeuralNetStrategy()
        try:
            type(strategy).use_learnable_cat_embeddings = False
            result = strategy.build_pipeline(
                base_pipeline=None,
                cat_features=["cat1"],
                category_encoder=OrdinalEncoder(),
                imputer=SimpleImputer(),
                scaler=StandardScaler(),
            )
            step_names = [name for name, _ in result.steps]
            assert "ce" in step_names
            assert "imp" in step_names
            assert "scaler" in step_names
        finally:
            type(strategy).use_learnable_cat_embeddings = True


class TestLinearModelStrategy:
    """Tests for LinearModelStrategy."""

    def test_cache_key(self):
        """Test linear strategy cache key."""
        strategy = LinearModelStrategy()
        assert strategy.cache_key == "linear"

    def test_requires_scaling_true(self):
        """Test linear models require scaling."""
        strategy = LinearModelStrategy()
        assert strategy.requires_scaling is True

    def test_requires_encoding_true(self):
        """Test linear models require encoding."""
        strategy = LinearModelStrategy()
        assert strategy.requires_encoding is True

    def test_requires_imputation_true(self):
        """Test linear models require imputation."""
        strategy = LinearModelStrategy()
        assert strategy.requires_imputation is True

    def test_same_scaling_imputation_as_neural(self):
        """Linear shares neural's scaling + imputation requirements. Encoding now DIFFERS: neural defaults to learnable cat embeddings
        (requires_encoding False) while linear still target-encodes (True), so they are compared separately."""
        linear = LinearModelStrategy()
        neural = NeuralNetStrategy()

        assert linear.requires_scaling == neural.requires_scaling
        assert linear.requires_imputation == neural.requires_imputation
        assert linear.requires_encoding is True
        assert neural.requires_encoding is False

    def test_different_cache_key_from_neural(self):
        """Test that linear and neural have different cache keys."""
        linear = LinearModelStrategy()
        neural = NeuralNetStrategy()

        assert linear.cache_key != neural.cache_key


# =============================================================================
# Strategy Registry Tests
# =============================================================================


class TestModelStrategies:
    """Tests for MODEL_STRATEGIES registry."""

    def test_tree_models_registered(self):
        """Test tree models are in registry."""
        tree_models = ["cb", "lgb", "xgb"]
        for model in tree_models:
            assert model in MODEL_STRATEGIES
            assert isinstance(MODEL_STRATEGIES[model], TreeModelStrategy)

    def test_hgb_models_registered(self):
        """Test HGB models are in registry."""
        hgb_models = ["hgb"]
        for model in hgb_models:
            assert model in MODEL_STRATEGIES
            assert isinstance(MODEL_STRATEGIES[model], HGBStrategy)

    def test_neural_models_registered(self):
        """Test neural models are in registry."""
        neural_models = ["mlp", "ngb"]
        for model in neural_models:
            assert model in MODEL_STRATEGIES
            assert isinstance(MODEL_STRATEGIES[model], NeuralNetStrategy)

    def test_linear_models_registered(self):
        """Test linear models are in registry."""
        linear_models = ["linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd", "logistic"]
        for model in linear_models:
            assert model in MODEL_STRATEGIES
            assert isinstance(MODEL_STRATEGIES[model], LinearModelStrategy)


class TestGetStrategy:
    """Tests for get_strategy function."""

    @pytest.mark.parametrize(
        "model_name,expected_strategy",
        [
            ("cb", "TreeModelStrategy"),
            ("hgb", "HGBStrategy"),
            ("mlp", "NeuralNetStrategy"),
            ("ridge", "LinearModelStrategy"),
        ],
        ids=["cb_tree", "hgb_hgb", "mlp_neural", "ridge_linear"],
    )
    def test_returns_correct_strategy_for_model(self, model_name, expected_strategy):
        """get_strategy returns the strategy class registered for the model family."""
        strategy_classes = {
            "TreeModelStrategy": TreeModelStrategy,
            "HGBStrategy": HGBStrategy,
            "NeuralNetStrategy": NeuralNetStrategy,
            "LinearModelStrategy": LinearModelStrategy,
        }
        strategy = get_strategy(model_name)
        assert isinstance(strategy, strategy_classes[expected_strategy])

    def test_case_insensitive(self):
        """Test get_strategy is case insensitive."""
        assert isinstance(get_strategy("CB"), TreeModelStrategy)
        assert isinstance(get_strategy("CATBOOST"), TreeModelStrategy)
        assert isinstance(get_strategy("Lgb"), TreeModelStrategy)

    def test_unknown_model_returns_tree_with_warning(self):
        """Test unknown model defaults to tree with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy = get_strategy("unknown_model")

            assert isinstance(strategy, TreeModelStrategy)
            assert len(w) == 1
            assert "Unknown model" in str(w[0].message)


# CACHE-P1-2: TestGetCacheKey class removed alongside the deleted helper. The
# underlying behaviour is still exercised by TestTreeModelStrategy etc. which
# assert ``get_strategy(model_name).cache_key`` directly.


# =============================================================================
# PipelineCache Tests
# =============================================================================


class TestPipelineCache:
    """Tests for PipelineCache class."""

    def test_init_empty(self):
        """Test cache initializes empty."""
        cache = PipelineCache()
        assert not cache.has("tree")
        assert cache.get("tree") is None

    def test_set_and_get(self):
        """Test setting and getting cached DataFrames."""
        cache = PipelineCache()
        train_df = MagicMock()
        val_df = MagicMock()
        test_df = MagicMock()

        cache.set("tree", train_df, val_df, test_df)

        result = cache.get("tree")
        assert result == (train_df, val_df, test_df)

    def test_has_after_set(self):
        """Test has returns True after setting."""
        cache = PipelineCache()
        cache.set("neural", MagicMock(), MagicMock(), MagicMock())

        assert cache.has("neural")
        assert not cache.has("tree")

    def test_multiple_keys(self):
        """Test cache with multiple keys."""
        cache = PipelineCache()

        train1, val1, test1 = MagicMock(), MagicMock(), MagicMock()
        train2, val2, test2 = MagicMock(), MagicMock(), MagicMock()

        cache.set("tree", train1, val1, test1)
        cache.set("neural", train2, val2, test2)

        assert cache.get("tree") == (train1, val1, test1)
        assert cache.get("neural") == (train2, val2, test2)

    def test_clear(self):
        """Test clearing the cache."""
        cache = PipelineCache()
        cache.set("tree", MagicMock(), MagicMock(), MagicMock())
        cache.set("neural", MagicMock(), MagicMock(), MagicMock())

        cache.clear()

        assert not cache.has("tree")
        assert not cache.has("neural")

    def test_overwrite_key(self):
        """Test overwriting existing key."""
        cache = PipelineCache()
        old_train = MagicMock()
        new_train = MagicMock()

        cache.set("tree", old_train, MagicMock(), MagicMock())
        cache.set("tree", new_train, MagicMock(), MagicMock())

        result = cache.get("tree")
        assert result[0] is new_train

    def test_get_nonexistent_returns_none(self):
        """Test getting nonexistent key returns None."""
        cache = PipelineCache()
        assert cache.get("nonexistent") is None


# =============================================================================
# Pipeline Building Tests
# =============================================================================


class TestBuildPipelineIntegration:
    """Integration tests for build_pipeline method."""

    def test_base_pipeline_only_returns_base(self):
        """Test with only base pipeline and no other components."""
        strategy = NeuralNetStrategy()
        base = Pipeline([("feat_sel", MagicMock())])

        result = strategy.build_pipeline(
            base_pipeline=base,
            cat_features=[],  # No categoricals
            category_encoder=None,
            imputer=None,
            scaler=None,
        )

        # Should return base pipeline unchanged
        assert result is base

    def test_all_components_added_in_order(self):
        """Test all components are added in correct order."""
        strategy = LinearModelStrategy()
        base = Pipeline([("base_step", MagicMock())])
        encoder = OrdinalEncoder()
        imputer = SimpleImputer()
        scaler = StandardScaler()

        result = strategy.build_pipeline(
            base_pipeline=base,
            cat_features=["cat1"],
            category_encoder=encoder,
            imputer=imputer,
            scaler=scaler,
        )

        assert isinstance(result, Pipeline)
        step_names = [name for name, _ in result.steps]
        # Order: ce (cat-encode), pre (selector), [inf_to_nan,] imp, scaler, [optional to_float32 tail]. The encoder runs
        # BEFORE the feature selector for a requires_encoding strategy so the selector's numeric estimator never sees raw
        # string cats ("could not convert string to float: 'C'"). The ``inf_to_nan`` step sits before the imputer so inf
        # -> NaN gets filled (SimpleImputer only handles NaN). The trailing 'to_float32' keeps linear inputs float32.
        assert step_names[:3] == ["ce", "pre", "inf_to_nan"] or step_names[:3] == ["ce", "pre", "imp"]
        if "inf_to_nan" in step_names:
            assert step_names[:5] == ["ce", "pre", "inf_to_nan", "imp", "scaler"]
            assert step_names[5:] in ([], ["to_float32"])
        else:
            assert step_names[:4] == ["ce", "pre", "imp", "scaler"]
            assert step_names[4:] in ([], ["to_float32"])

    def test_partial_components(self):
        """Test with only some components provided."""
        strategy = LinearModelStrategy()
        scaler = StandardScaler()

        result = strategy.build_pipeline(
            base_pipeline=None,
            cat_features=[],  # No categoricals
            category_encoder=None,
            imputer=None,
            scaler=scaler,
        )

        assert isinstance(result, Pipeline)
        step_names = [name for name, _ in result.steps]
        assert "scaler" in step_names
        assert "ce" not in step_names
        assert "imp" not in step_names
