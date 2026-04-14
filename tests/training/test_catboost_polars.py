"""Tests for native Polars DataFrame support in ML estimators.

Verifies that CatBoost and HGB can train directly on Polars DataFrames
(no pandas conversion) with early stopping on a separate Polars validation set.
"""

import numpy as np
import polars as pl
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

RANDOM_SEED = 42
N_TRAIN = 200
N_VAL = 50
ITERATIONS = 20
EARLY_STOPPING_ROUNDS = 5

FEATURE_COLS = ["num_feat", "cat_feat", "text_feat", "emb_feat"]
CAT_FEATURES = ["cat_feat"]
TEXT_FEATURES = ["text_feat"]
EMBEDDING_FEATURES = ["emb_feat"]


def _make_data(n: int, rng: np.random.Generator) -> pl.DataFrame:
    """Build a Polars DataFrame with numeric, categorical, text, and embedding columns."""
    return pl.DataFrame({
        "num_feat": rng.standard_normal(n),
        "cat_feat": rng.choice(["a", "b", "c", "d"], size=n),
        "text_feat": rng.choice(
            ["the quick brown fox", "lazy dog sleeps", "hello world program", "data science rocks"],
            size=n,
        ),
        "emb_feat": [rng.standard_normal(3).tolist() for _ in range(n)],
        "target_cls": rng.integers(0, 2, size=n),
        "target_reg": rng.standard_normal(n),
    })


@pytest.fixture()
def data():
    rng = np.random.default_rng(RANDOM_SEED)
    train = _make_data(N_TRAIN, rng)
    val = _make_data(N_VAL, rng)
    return train, val


def _make_pool(df: pl.DataFrame, target_col: str) -> Pool:
    return Pool(
        data=df.select(FEATURE_COLS),
        label=df[target_col].to_numpy(),
        cat_features=CAT_FEATURES,
        text_features=TEXT_FEATURES,
        embedding_features=EMBEDDING_FEATURES,
    )


class TestCatBoostPolarsClassification:
    """CatBoostClassifier trained on Polars DataFrames."""

    def test_train_with_early_stopping(self, data):
        train_df, val_df = data
        train_pool = _make_pool(train_df, "target_cls")
        val_pool = _make_pool(val_df, "target_cls")

        model = CatBoostClassifier(
            iterations=ITERATIONS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_seed=RANDOM_SEED,
            verbose=0,
        )
        model.fit(train_pool, eval_set=val_pool)

        assert model.is_fitted()
        assert model.tree_count_ <= ITERATIONS

    def test_predict_returns_correct_shape(self, data):
        train_df, val_df = data
        train_pool = _make_pool(train_df, "target_cls")
        val_pool = _make_pool(val_df, "target_cls")

        model = CatBoostClassifier(
            iterations=ITERATIONS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_seed=RANDOM_SEED,
            verbose=0,
        )
        model.fit(train_pool, eval_set=val_pool)

        preds = model.predict(val_pool)
        probas = model.predict_proba(val_pool)

        assert preds.shape == (N_VAL,)
        assert probas.shape == (N_VAL, 2)

    def test_feature_importance_available(self, data):
        train_df, val_df = data
        train_pool = _make_pool(train_df, "target_cls")
        val_pool = _make_pool(val_df, "target_cls")

        model = CatBoostClassifier(
            iterations=ITERATIONS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_seed=RANDOM_SEED,
            verbose=0,
        )
        model.fit(train_pool, eval_set=val_pool)

        importances = model.get_feature_importance()
        assert len(importances) == len(FEATURE_COLS)


class TestCatBoostPolarsRegression:
    """CatBoostRegressor trained on Polars DataFrames."""

    def test_train_with_early_stopping(self, data):
        train_df, val_df = data
        train_pool = _make_pool(train_df, "target_reg")
        val_pool = _make_pool(val_df, "target_reg")

        model = CatBoostRegressor(
            iterations=ITERATIONS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_seed=RANDOM_SEED,
            verbose=0,
        )
        model.fit(train_pool, eval_set=val_pool)

        assert model.is_fitted()
        assert model.tree_count_ <= ITERATIONS

    def test_predict_returns_correct_shape(self, data):
        train_df, val_df = data
        train_pool = _make_pool(train_df, "target_reg")
        val_pool = _make_pool(val_df, "target_reg")

        model = CatBoostRegressor(
            iterations=ITERATIONS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_seed=RANDOM_SEED,
            verbose=0,
        )
        model.fit(train_pool, eval_set=val_pool)

        preds = model.predict(val_pool)
        assert preds.shape == (N_VAL,)


def _gpu_available() -> bool:
    """Check if CatBoost can use the GPU."""
    try:
        probe = CatBoostClassifier(iterations=1, task_type="GPU", devices="0", verbose=0)
        rng = np.random.default_rng(0)
        X = pl.DataFrame({"a": rng.standard_normal(20)})
        y = rng.integers(0, 2, size=20)
        probe.fit(Pool(X, label=y))
        return True
    except Exception:
        return False


_has_gpu = _gpu_available()


@pytest.mark.skipif(not _has_gpu, reason="No GPU available for CatBoost")
class TestCatBoostPolarsGPUClassification:
    """CatBoostClassifier on GPU with Polars DataFrames."""

    def test_train_gpu_classification(self, data):
        train_df, val_df = data
        train_pool = _make_pool(train_df, "target_cls")
        val_pool = _make_pool(val_df, "target_cls")

        model = CatBoostClassifier(
            iterations=ITERATIONS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            task_type="GPU",
            devices="0",
            random_seed=RANDOM_SEED,
            verbose=0,
        )
        model.fit(train_pool, eval_set=val_pool)

        assert model.is_fitted()
        assert model.tree_count_ <= ITERATIONS
        preds = model.predict_proba(val_pool)
        assert preds.shape == (N_VAL, 2)


@pytest.mark.skipif(not _has_gpu, reason="No GPU available for CatBoost")
class TestCatBoostPolarsGPURegression:
    """CatBoostRegressor on GPU with Polars DataFrames."""

    def test_train_gpu_regression(self, data):
        train_df, val_df = data
        train_pool = _make_pool(train_df, "target_reg")
        val_pool = _make_pool(val_df, "target_reg")

        model = CatBoostRegressor(
            iterations=ITERATIONS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            task_type="GPU",
            devices="0",
            random_seed=RANDOM_SEED,
            verbose=0,
        )
        model.fit(train_pool, eval_set=val_pool)

        assert model.is_fitted()
        assert model.tree_count_ <= ITERATIONS
        preds = model.predict(val_pool)
        assert preds.shape == (N_VAL,)


class TestPolarsPoolCreation:
    """Verify Pool accepts Polars DataFrames directly (no pandas conversion)."""

    def test_pool_from_polars_no_conversion(self, data):
        train_df, _ = data
        features = train_df.select(FEATURE_COLS)
        assert isinstance(features, pl.DataFrame)

        pool = Pool(
            data=features,
            label=train_df["target_cls"].to_numpy(),
            cat_features=CAT_FEATURES,
            text_features=TEXT_FEATURES,
            embedding_features=EMBEDDING_FEATURES,
        )
        assert pool.num_row() == N_TRAIN
        assert pool.num_col() == len(FEATURE_COLS)

    def test_pool_shape_matches_input(self, data):
        train_df, val_df = data
        train_pool = _make_pool(train_df, "target_reg")
        val_pool = _make_pool(val_df, "target_reg")

        assert train_pool.num_row() == N_TRAIN
        assert val_pool.num_row() == N_VAL


# ---------------------------------------------------------------------------
# HGB (HistGradientBoosting) — numeric-only Polars support
# ---------------------------------------------------------------------------

def _make_numeric_data(n: int, rng: np.random.Generator) -> pl.DataFrame:
    """Build a Polars DataFrame with numeric-only features (HGB can't handle string cols)."""
    return pl.DataFrame({
        "num_a": rng.standard_normal(n),
        "num_b": rng.standard_normal(n),
        "num_c": rng.standard_normal(n),
        "target_cls": rng.integers(0, 2, size=n),
        "target_reg": rng.standard_normal(n),
    })


HGB_FEATURE_COLS = ["num_a", "num_b", "num_c"]


@pytest.fixture()
def numeric_data():
    rng = np.random.default_rng(RANDOM_SEED)
    train = _make_numeric_data(N_TRAIN, rng)
    val = _make_numeric_data(N_VAL, rng)
    return train, val


class TestHGBPolarsClassification:
    """HistGradientBoostingClassifier trained on Polars DataFrames (numeric only)."""

    def test_train_with_early_stopping(self, numeric_data):
        train_df, val_df = numeric_data
        X_train = train_df.select(HGB_FEATURE_COLS)
        y_train = train_df["target_cls"].to_numpy()
        X_val = val_df.select(HGB_FEATURE_COLS)
        y_val = val_df["target_cls"].to_numpy()

        model = HistGradientBoostingClassifier(
            max_iter=ITERATIONS,
            early_stopping=True,
            validation_fraction=None,
            n_iter_no_change=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        assert hasattr(model, "n_iter_")
        assert model.n_iter_ <= ITERATIONS

    def test_predict_returns_correct_shape(self, numeric_data):
        train_df, val_df = numeric_data
        X_train = train_df.select(HGB_FEATURE_COLS)
        y_train = train_df["target_cls"].to_numpy()

        model = HistGradientBoostingClassifier(max_iter=ITERATIONS, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        preds = model.predict(val_df.select(HGB_FEATURE_COLS))
        probas = model.predict_proba(val_df.select(HGB_FEATURE_COLS))

        assert preds.shape == (N_VAL,)
        assert probas.shape == (N_VAL, 2)


class TestXGBoostStrategyPreparePolars:
    """Unit tests for XGBoostStrategy.prepare_polars_dataframe."""

    def test_string_to_categorical(self):
        from mlframe.training.strategies import XGBoostStrategy
        strategy = XGBoostStrategy()
        df = pl.DataFrame({"cat": ["a", "b", "c", "a"], "num": [1.0, 2.0, 3.0, 4.0]})
        result = strategy.prepare_polars_dataframe(df, ["cat"])
        assert result["cat"].dtype == pl.Categorical

    def test_high_cardinality_stays_categorical(self):
        """Unlike HGB, XGBoost has no 255 cardinality limit."""
        from mlframe.training.strategies import XGBoostStrategy
        strategy = XGBoostStrategy()
        cats = [f"cat_{i}" for i in range(500)]
        df = pl.DataFrame({"cat": cats})
        result = strategy.prepare_polars_dataframe(df, ["cat"])
        assert result["cat"].dtype == pl.Categorical  # NOT ordinal encoded

    def test_already_categorical_unchanged(self):
        from mlframe.training.strategies import XGBoostStrategy
        strategy = XGBoostStrategy()
        df = pl.DataFrame({"cat": ["a", "b", "c"]}).with_columns(pl.col("cat").cast(pl.Categorical))
        result = strategy.prepare_polars_dataframe(df, ["cat"])
        assert result["cat"].dtype == pl.Categorical

    def test_numeric_passthrough(self):
        from mlframe.training.strategies import XGBoostStrategy
        strategy = XGBoostStrategy()
        df = pl.DataFrame({"num": [1.0, 2.0, 3.0]})
        result = strategy.prepare_polars_dataframe(df, [])
        assert result.equals(df)


class TestXGBoostPolarsClassification:
    """XGBoost trained on Polars DataFrames with categorical features."""

    def test_train_with_categorical(self):
        from xgboost import XGBClassifier
        from mlframe.training.strategies import XGBoostStrategy
        rng = np.random.default_rng(RANDOM_SEED)

        n = 200
        df = pl.DataFrame({
            "num_a": rng.standard_normal(n),
            "cat_a": rng.choice(["x", "y", "z"], size=n),
            "target": rng.integers(0, 2, size=n),
        })

        strategy = XGBoostStrategy()
        df = strategy.prepare_polars_dataframe(df, ["cat_a"])
        assert df["cat_a"].dtype == pl.Categorical

        X = df.select(["num_a", "cat_a"])
        y = df["target"].to_numpy()

        model = XGBClassifier(
            n_estimators=ITERATIONS,
            enable_categorical=True,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (n,)


class TestHGBStrategyPreparePolars:
    """Unit tests for HGBStrategy.prepare_polars_dataframe."""

    def test_string_to_categorical_low_cardinality(self):
        from mlframe.training.strategies import HGBStrategy
        strategy = HGBStrategy()
        df = pl.DataFrame({"cat": ["a", "b", "c", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = strategy.prepare_polars_dataframe(df, ["cat"])
        assert result["cat"].dtype == pl.Categorical

    def test_string_to_ordinal_high_cardinality(self):
        from mlframe.training.strategies import HGBStrategy
        strategy = HGBStrategy()
        # 300 unique values > 255 limit
        cats = [f"cat_{i}" for i in range(300)]
        df = pl.DataFrame({"cat": cats, "num": list(range(300))})
        result = strategy.prepare_polars_dataframe(df, ["cat"])
        assert result["cat"].dtype == pl.UInt32

    def test_already_categorical_low_cardinality_unchanged(self):
        from mlframe.training.strategies import HGBStrategy
        strategy = HGBStrategy()
        df = pl.DataFrame({"cat": ["a", "b", "c"]}).with_columns(pl.col("cat").cast(pl.Categorical))
        result = strategy.prepare_polars_dataframe(df, ["cat"])
        assert result["cat"].dtype == pl.Categorical

    def test_already_categorical_high_cardinality_ordinal(self):
        from mlframe.training.strategies import HGBStrategy
        strategy = HGBStrategy()
        cats = [f"cat_{i}" for i in range(300)]
        df = pl.DataFrame({"cat": cats}).with_columns(pl.col("cat").cast(pl.Categorical))
        result = strategy.prepare_polars_dataframe(df, ["cat"])
        assert result["cat"].dtype == pl.UInt32

    def test_no_cat_features_passthrough(self):
        from mlframe.training.strategies import HGBStrategy
        strategy = HGBStrategy()
        df = pl.DataFrame({"num": [1.0, 2.0, 3.0]})
        result = strategy.prepare_polars_dataframe(df, [])
        assert result.equals(df)

    def test_missing_cat_column_ignored(self):
        from mlframe.training.strategies import HGBStrategy
        strategy = HGBStrategy()
        df = pl.DataFrame({"num": [1.0, 2.0, 3.0]})
        result = strategy.prepare_polars_dataframe(df, ["nonexistent"])
        assert result.equals(df)

    def test_boundary_255_categories(self):
        from mlframe.training.strategies import HGBStrategy
        strategy = HGBStrategy()
        cats = [f"c{i}" for i in range(255)]
        df = pl.DataFrame({"cat": cats})
        result = strategy.prepare_polars_dataframe(df, ["cat"])
        assert result["cat"].dtype == pl.Categorical  # exactly 255 = OK

    def test_boundary_256_categories(self):
        from mlframe.training.strategies import HGBStrategy
        strategy = HGBStrategy()
        cats = [f"c{i}" for i in range(256)]
        df = pl.DataFrame({"cat": cats})
        result = strategy.prepare_polars_dataframe(df, ["cat"])
        assert result["cat"].dtype == pl.UInt32  # 256 > 255 = ordinal encode


class TestHGBPolarsRegression:
    """HistGradientBoostingRegressor trained on Polars DataFrames (numeric only)."""

    def test_train_with_early_stopping(self, numeric_data):
        train_df, val_df = numeric_data
        X_train = train_df.select(HGB_FEATURE_COLS)
        y_train = train_df["target_reg"].to_numpy()
        X_val = val_df.select(HGB_FEATURE_COLS)
        y_val = val_df["target_reg"].to_numpy()

        model = HistGradientBoostingRegressor(
            max_iter=ITERATIONS,
            early_stopping=True,
            validation_fraction=None,
            n_iter_no_change=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        assert hasattr(model, "n_iter_")
        assert model.n_iter_ <= ITERATIONS

    def test_predict_returns_correct_shape(self, numeric_data):
        train_df, val_df = numeric_data
        X_train = train_df.select(HGB_FEATURE_COLS)
        y_train = train_df["target_reg"].to_numpy()

        model = HistGradientBoostingRegressor(max_iter=ITERATIONS, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        preds = model.predict(val_df.select(HGB_FEATURE_COLS))
        assert preds.shape == (N_VAL,)
