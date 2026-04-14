"""
Integration test: train CatBoost on a synthetic Polars DataFrame with mixed dtypes
matching a real production dataset shape.

Real dataset dtypes:
  Boolean(10), Categorical(68), Datetime(1), Float32(38), Float64(416),
  Int16(14), Int64(2), Int8(27)
  + classification target derived from a numeric column via threshold.

Total: ~576 feature columns + target source + metadata columns, 100k rows.
"""

import numpy as np
import polars as pl
import pytest

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.configs import (
    ModelHyperparamsConfig,
    PolarsPipelineConfig,
    TrainingBehaviorConfig,
    TrainingSplitConfig,
)
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor


def _make_synthetic_mixed_df(n_rows: int = 10_000, seed: int = 42) -> pl.DataFrame:
    """Build a synthetic Polars DF mirroring the real production dataset dtypes."""
    rng = np.random.default_rng(seed)

    cols: dict[str, pl.Series] = {}

    # --- Boolean(10) ---
    for i in range(10):
        cols[f"bool_{i}"] = pl.Series(f"bool_{i}", rng.choice([True, False], n_rows))

    # --- Categorical(68) ---
    cat_pools = [
        rng.choice([f"cat{j}" for j in range(k)], n_rows)
        for k in rng.integers(3, 30, size=68)
    ]
    for i, pool in enumerate(cat_pools):
        cols[f"cat_{i}"] = pl.Series(f"cat_{i}", pool).cast(pl.Categorical)

    # --- Datetime(1) ---
    base_ts = np.datetime64("2024-01-01")
    offsets = rng.integers(0, 365 * 24 * 3600, n_rows).astype("timedelta64[s]")
    cols["job_posted_at"] = pl.Series("job_posted_at", (base_ts + offsets).astype("datetime64[us]"))

    # --- Float32(38) ---
    for i in range(38):
        vals = rng.standard_normal(n_rows).astype(np.float32)
        # sprinkle ~1% NaN
        vals[rng.random(n_rows) < 0.01] = np.nan
        cols[f"f32_{i}"] = pl.Series(f"f32_{i}", vals)

    # --- Float64(416) ---
    for i in range(416):
        vals = rng.standard_normal(n_rows)
        vals[rng.random(n_rows) < 0.01] = np.nan
        cols[f"f64_{i}"] = pl.Series(f"f64_{i}", vals)

    # --- Int16(14) ---
    for i in range(14):
        cols[f"i16_{i}"] = pl.Series(f"i16_{i}", rng.integers(-500, 500, n_rows, dtype=np.int16))

    # --- Int64(2) ---
    for i in range(2):
        cols[f"i64_{i}"] = pl.Series(f"i64_{i}", rng.integers(0, 1_000_000, n_rows, dtype=np.int64))

    # --- Int8(27) ---
    for i in range(27):
        cols[f"i8_{i}"] = pl.Series(f"i8_{i}", rng.integers(-50, 50, n_rows, dtype=np.int8))

    # --- Target source column (mimics cl_act_total_hired) ---
    cols["cl_act_total_hired"] = pl.Series(
        "cl_act_total_hired", rng.integers(0, 5, n_rows, dtype=np.int16)
    )

    # --- Metadata columns to drop ---
    cols["uid"] = pl.Series("uid", np.arange(n_rows, dtype=np.int64))
    cols["job_status"] = pl.Series("job_status", rng.choice(["open", "closed", "filled"], n_rows)).cast(pl.Categorical)
    cols["cl_id"] = pl.Series("cl_id", rng.integers(1, 10_000, n_rows, dtype=np.int64))

    return pl.DataFrame(cols)


class TestMixedDtypesTraining:
    """Train CatBoost on a realistic mixed-dtype Polars DataFrame."""

    @pytest.fixture(scope="class")
    def synthetic_df(self):
        return _make_synthetic_mixed_df()

    def test_catboost_trains_on_mixed_dtypes(self, synthetic_df, tmp_path):
        """
        Equivalent of the user's old calling code, translated to the new API.

        Old API used: config_params_override, control_params_override
        New API uses: hyperparams_config, behavior_config (Pydantic or dict)
        """
        ft_extractor = SimpleFeaturesAndTargetsExtractor(
            classification_targets=["cl_act_total_hired"],
            classification_lower_thresholds=dict(cl_act_total_hired=1),
            ts_field="job_posted_at",
            columns_to_drop={"uid", "job_posted_at", "job_status", "cl_id"},
            verbose=1,
        )

        mlframe_models, metadata = train_mlframe_models_suite(
            df=synthetic_df,
            target_name="H",
            model_name="prod_jobsdetails",
            features_and_targets_extractor=ft_extractor,
            mlframe_models=["cb"],
            init_common_params={"show_perf_chart": False, "show_fi": False},
            use_ordinary_models=True,
            pipeline_config=PolarsPipelineConfig(
                use_polarsds_pipeline=False,
                categorical_encoding=None,
                scaler_name=None,
                imputer_strategy=None,
            ),
            split_config=TrainingSplitConfig(
                shuffle_val=False,
                shuffle_test=False,
                test_size=0.1,
                val_size=0.1,
                wholeday_splitting=False,
            ),
            # --- NEW API (replaces config_params_override / control_params_override) ---
            hyperparams_config=ModelHyperparamsConfig(
                iterations=50,
                early_stopping_rounds=10,
                cb_kwargs={"task_type": "CPU"},
            ),
            behavior_config=TrainingBehaviorConfig(
                prefer_calibrated_classifiers=False,
            ),
            use_mlframe_ensembles=False,
            data_dir=str(tmp_path / "data"),
            verbose=True,
        )

        # Verify models were trained
        assert mlframe_models is not None
        assert len(mlframe_models) > 0
        # Verify metadata structure
        assert "columns" in metadata
        assert "pipeline" in metadata
